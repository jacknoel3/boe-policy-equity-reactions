from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import load_equity_index_from_consolidated


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

EXCEL_FILE = DATA_DIR / "measuring-monetary-policy-in-the-uk-the-ukmpesd.xlsx"
EQUITY_FILE = DATA_DIR / "uk_equity_indices_PX_LAST_19970101_20260312.csv"
MAX_HORIZON = 20
RETURN_LAGS = 5
CONF_Z = 1.96


def ensure_output_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def detect_shock_sheet(excel_file: Path) -> tuple[str, list[str]]:
    workbook = pd.ExcelFile(excel_file)
    sheet_names = workbook.sheet_names
    print("Workbook sheet names:")
    for sheet in sheet_names:
        print(f"  - {sheet}")

    selected_sheet = None
    for sheet in sheet_names:
        preview = pd.read_excel(excel_file, sheet_name=sheet, nrows=5)
        normalized = {str(col).strip().lower() for col in preview.columns}
        if {"target", "path", "qe"}.issubset(normalized):
            selected_sheet = sheet
            break

    if selected_sheet is None:
        raise ValueError("Could not identify a shock sheet containing Target, Path, and QE columns.")

    print(f"\nSelected shock sheet: {selected_sheet}")
    return selected_sheet, sheet_names


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    shock_sheet, _ = detect_shock_sheet(EXCEL_FILE)

    shocks_raw = pd.read_excel(EXCEL_FILE, sheet_name=shock_sheet)
    print("\nRaw shock columns:")
    print(list(shocks_raw.columns))

    equity = load_equity_index_from_consolidated(EQUITY_FILE, index_name="ftse100").copy()
    equity["date"] = equity["date"].dt.date

    print("\nEquity sample diagnostics:")
    print(f"  - Observations after cleaning: {len(equity):,}")
    print(f"  - Date range: {equity['date'].min()} to {equity['date'].max()}")
    print("  - Return summary:")
    print(equity["log_return"].describe().to_string())

    return shocks_raw, equity


def prepare_shocks(shocks_raw: pd.DataFrame) -> pd.DataFrame:
    normalized_map = {str(col).strip().lower(): col for col in shocks_raw.columns}

    datetime_col = normalized_map.get("datetime")
    if datetime_col is None:
        datetime_candidates = [col for col in shocks_raw.columns if "date" in str(col).lower() or "time" in str(col).lower()]
        if not datetime_candidates:
            raise ValueError("Could not detect the shock datetime column.")
        datetime_col = datetime_candidates[0]

    required = ["target", "path", "qe"]
    missing = [col for col in required if col not in normalized_map]
    if missing:
        raise ValueError(f"Missing required shock columns: {missing}")

    shocks = shocks_raw[[datetime_col, normalized_map["target"], normalized_map["path"], normalized_map["qe"]]].copy()
    shocks.columns = ["datetime", "target", "path", "qe"]
    shocks["datetime"] = pd.to_datetime(shocks["datetime"], errors="coerce")
    shocks = shocks.dropna(subset=["datetime"])
    shocks["date"] = shocks["datetime"].dt.date

    for col in ["target", "path", "qe"]:
        shocks[col] = pd.to_numeric(shocks[col], errors="coerce")

    shocks = shocks.drop(columns=["datetime"])
    shocks = shocks.groupby("date", as_index=False)[["target", "path", "qe"]].sum(min_count=1)
    shocks[["target", "path", "qe"]] = shocks[["target", "path", "qe"]].fillna(0.0)

    print("\nPrepared shock data diagnostics:")
    print(f"  - Shock-day observations: {len(shocks):,}")
    print(f"  - Date range: {shocks['date'].min()} to {shocks['date'].max()}")
    print("  - Shock summary:")
    print(shocks[["target", "path", "qe"]].describe().to_string())

    return shocks


def merge_data(equity: pd.DataFrame, shocks: pd.DataFrame) -> pd.DataFrame:
    shock_sample_start = pd.to_datetime(shocks["date"]).min()
    buffer_start = (pd.Timestamp(shock_sample_start) - pd.tseries.offsets.BDay(RETURN_LAGS)).date()
    equity = equity[pd.to_datetime(equity["date"]) >= pd.Timestamp(buffer_start)].copy()
    equity["shock_sample_start"] = shock_sample_start.date()
    equity["in_estimation_sample"] = pd.to_datetime(equity["date"]) >= pd.Timestamp(shock_sample_start)

    print(f"\nKeeping lag buffer from: {buffer_start}")
    print(f"  - Estimation sample starts: {shock_sample_start}")
    print(f"  - Trading-day observations retained: {len(equity):,}")

    merged = equity.merge(shocks, on="date", how="left")
    for col in ["target", "path", "qe"]:
        merged[col] = merged[col].fillna(0.0)

    shock_day_mask = merged[["target", "path", "qe"]].ne(0).any(axis=1)
    print("\nMerged data diagnostics:")
    print(f"  - Total observations: {len(merged):,}")
    print(f"  - Shock days: {int(shock_day_mask.sum()):,}")
    print(f"  - Non-missing returns: {int(merged['log_return'].notna().sum()):,}")

    return merged


def create_asymmetric_shocks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for shock in ["target", "path", "qe"]:
        df[f"{shock}_tightening"] = np.maximum(df[shock], 0.0)
        df[f"{shock}_easing"] = -np.minimum(df[shock], 0.0)
    return df


def create_lp_variables(df: pd.DataFrame, max_horizon: int = MAX_HORIZON, return_lags: int = RETURN_LAGS) -> pd.DataFrame:
    df = df.copy()

    for lag in range(1, return_lags + 1):
        df[f"log_return_lag{lag}"] = df["log_return"].shift(lag)

    for horizon in range(max_horizon + 1):
        cumulative_return = sum(df["log_return"].shift(-step) for step in range(horizon + 1))
        df[f"cum_return_h{horizon}"] = cumulative_return

    return df


def run_local_projection(df: pd.DataFrame, shock_name: str, max_horizon: int = MAX_HORIZON, return_lags: int = RETURN_LAGS) -> pd.DataFrame:
    results = []
    tightening_var = f"{shock_name}_tightening"
    easing_var = f"{shock_name}_easing"
    lag_vars = [f"log_return_lag{lag}" for lag in range(1, return_lags + 1)]

    for horizon in range(max_horizon + 1):
        y_var = f"cum_return_h{horizon}"
        regression_df = df[["in_estimation_sample", y_var, tightening_var, easing_var] + lag_vars].dropna().copy()
        regression_df = regression_df[regression_df["in_estimation_sample"]].copy()

        if regression_df.empty:
            print(f"Skipping {shock_name} horizon {horizon}: no usable observations.")
            continue

        y = regression_df[y_var]
        x = sm.add_constant(regression_df[[tightening_var, easing_var] + lag_vars], has_constant="add")
        model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": horizon + 1})

        beta_tightening = model.params[tightening_var]
        se_tightening = model.bse[tightening_var]
        beta_easing = model.params[easing_var]
        se_easing = model.bse[easing_var]
        ci_tightening_low = beta_tightening - CONF_Z * se_tightening
        ci_tightening_high = beta_tightening + CONF_Z * se_tightening
        ci_easing_low = beta_easing - CONF_Z * se_easing
        ci_easing_high = beta_easing + CONF_Z * se_easing

        restriction = np.zeros((1, len(model.params)))
        param_names = list(model.params.index)
        restriction[0, param_names.index(tightening_var)] = 1.0
        restriction[0, param_names.index(easing_var)] = -1.0
        asymmetry_test = model.t_test(restriction)
        asymmetry_pvalue = float(np.asarray(asymmetry_test.pvalue).squeeze())

        results.append(
            {
                "shock": shock_name,
                "horizon": horizon,
                "beta_tightening": beta_tightening,
                "se_tightening": se_tightening,
                "ci_lower_tightening": ci_tightening_low,
                "ci_upper_tightening": ci_tightening_high,
                "beta_easing": beta_easing,
                "se_easing": se_easing,
                "ci_lower_easing": ci_easing_low,
                "ci_upper_easing": ci_easing_high,
                "asymmetry_pvalue": asymmetry_pvalue,
                "nobs": int(model.nobs),
                "r_squared": model.rsquared,
            }
        )

    results_df = pd.DataFrame(results)
    print(f"\nLocal projection summary for {shock_name}:")
    print(results_df.head().to_string(index=False))
    return results_df


def plot_irf(results_df: pd.DataFrame, shock_name: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    horizons = results_df["horizon"]
    ax.plot(horizons, results_df["beta_tightening"], label="Tightening shock", color="tab:blue", linewidth=2)
    ax.fill_between(
        horizons,
        results_df["ci_lower_tightening"],
        results_df["ci_upper_tightening"],
        color="tab:blue",
        alpha=0.2,
    )

    ax.plot(horizons, results_df["beta_easing"], label="Easing shock", color="tab:red", linewidth=2)
    ax.fill_between(
        horizons,
        results_df["ci_lower_easing"],
        results_df["ci_upper_easing"],
        color="tab:red",
        alpha=0.2,
    )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    display_name = "QE" if shock_name == "qe" else shock_name.capitalize()
    ax.set_title(f"Local Projection IRFs: {display_name} Shock")
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Cumulative log return response")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    figure_path = FIGURES_DIR / f"irf_{shock_name}_v1.png"
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {figure_path}")


def main():
    ensure_output_dirs()

    print("Assumptions used in this baseline specification:")
    print("  - Shock sheet is identified by the presence of Target, Path, and QE columns.")
    print("  - FTSE date and price columns are detected programmatically.")
    print("  - Daily shocks are summed within date before merging to returns.")
    print("  - Missing shocks on equity trading days are set to zero.")
    print("  - Local projections use 5 lags of daily log returns and HAC standard errors with maxlags = h + 1.")

    shocks_raw, equity = load_data()
    shocks = prepare_shocks(shocks_raw)
    merged = merge_data(equity, shocks)
    merged = create_asymmetric_shocks(merged)
    merged = create_lp_variables(merged, max_horizon=MAX_HORIZON, return_lags=RETURN_LAGS)

    merged_output = OUTPUT_DIR / "lp_equity_monetary_shocks_merged_v1.csv"
    merged.to_csv(merged_output, index=False)
    print(f"\nSaved merged dataset: {merged_output}")

    for shock_name in ["target", "path", "qe"]:
        results_df = run_local_projection(merged, shock_name, max_horizon=MAX_HORIZON, return_lags=RETURN_LAGS)
        table_path = TABLES_DIR / f"lp_irf_{shock_name}_v1.csv"
        results_df.to_csv(table_path, index=False)
        print(f"Saved table: {table_path}")
        plot_irf(results_df, shock_name)


if __name__ == "__main__":
    main()
