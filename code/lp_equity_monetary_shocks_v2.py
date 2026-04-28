from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from utils import load_equity_index_from_consolidated


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output_v2"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

SHOCK_FILE = DATA_DIR / "measuring-monetary-policy-in-the-uk-the-ukmpesd.xlsx"
EQUITY_FILE = DATA_DIR / "uk_equity_indices_PX_LAST_19970101_20260312.csv"
INDEX_DISPLAY_NAMES = {
    "ftse100": "FTSE 100",
    "ftse250": "FTSE 250",
    "ftse_all_share": "FTSE All-Share",
}

SHOCK_NAMES = ["target", "path", "qe"]
MAX_HORIZON = 20
RETURN_LAGS = 5
CONF_Z = 1.96
QE_START_DATE = pd.Timestamp("2009-03-05")
SUPER_THURSDAY_START_DATE = pd.Timestamp("2015-08-06")
SELECTED_MAGNITUDE_HORIZONS = [0, 5, 10, 20]


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def get_index_display_name(index_name: str) -> str:
    return INDEX_DISPLAY_NAMES.get(index_name, index_name.upper())


def get_sample_display_name(sample_name: str) -> str:
    sample_display_names = {
        "full": "full",
        "pre_2009": "pre-2009 (before 2009-03-05 QE start)",
        "post_2009": "post-2009 (on/after 2009-03-05 QE start)",
        "pre_super_thursday": "pre-Super Thursday (before 2015-08-06)",
        "post_super_thursday": "post-Super Thursday (on/after 2015-08-06)",
    }
    return sample_display_names.get(sample_name, sample_name)


def load_shock_data(shock_file: Path) -> pd.DataFrame:
    shocks_raw = pd.read_excel(shock_file, sheet_name="factors")
    normalized_map = {str(col).strip().lower(): col for col in shocks_raw.columns}

    datetime_col = normalized_map.get("datetime")
    if datetime_col is None:
        raise ValueError("Could not detect the shock datetime column.")

    missing = [shock for shock in SHOCK_NAMES if shock not in normalized_map]
    if missing:
        raise ValueError(f"Missing required shock columns in the factors sheet: {missing}")

    shocks = shocks_raw[[datetime_col] + [normalized_map[shock] for shock in SHOCK_NAMES]].copy()
    shocks.columns = ["datetime"] + SHOCK_NAMES
    shocks["datetime"] = pd.to_datetime(shocks["datetime"], errors="coerce")
    shocks = shocks.dropna(subset=["datetime"])
    shocks["date"] = shocks["datetime"].dt.normalize()

    for shock in SHOCK_NAMES:
        shocks[shock] = pd.to_numeric(shocks[shock], errors="coerce")

    shocks = shocks.drop(columns=["datetime"])
    shocks = shocks.groupby("date", as_index=False)[SHOCK_NAMES].sum(min_count=1)
    shocks[SHOCK_NAMES] = shocks[SHOCK_NAMES].fillna(0.0)

    print("\nLoaded shock data")
    print(f"  - File: {shock_file.name}")
    print("  - Sheet: factors")
    print(f"  - Shock-day observations: {len(shocks):,}")
    print(f"  - Date range: {shocks['date'].min().date()} to {shocks['date'].max().date()}")
    for shock in SHOCK_NAMES:
        nonzero_days = int(shocks[shock].ne(0).sum())
        print(f"  - Non-zero {shock} days: {nonzero_days:,}")

    return shocks


def keep_lag_buffer_before_shock_sample(equity: pd.DataFrame, shocks: pd.DataFrame) -> pd.DataFrame:
    shock_sample_start = shocks["date"].min()
    buffer_start = shock_sample_start - pd.tseries.offsets.BDay(RETURN_LAGS)
    buffered = equity[equity["date"] >= buffer_start].copy()
    buffered["shock_sample_start"] = shock_sample_start
    buffered["in_estimation_sample"] = buffered["date"] >= shock_sample_start

    print(f"  - Keeping lag buffer from: {buffer_start.date()}")
    print(f"  - Estimation sample starts: {shock_sample_start.date()}")
    print(f"  - Trading-day observations retained: {len(buffered):,}")
    return buffered


def merge_equity_and_shocks(equity: pd.DataFrame, shocks: pd.DataFrame) -> pd.DataFrame:
    equity = keep_lag_buffer_before_shock_sample(equity, shocks)
    merged = equity.merge(shocks, on="date", how="left")
    merged[SHOCK_NAMES] = merged[SHOCK_NAMES].fillna(0.0)

    for shock in SHOCK_NAMES:
        merged[f"{shock}_tightening"] = np.maximum(merged[shock], 0.0)
        merged[f"{shock}_easing"] = -np.minimum(merged[shock], 0.0)

    for lag in range(1, RETURN_LAGS + 1):
        merged[f"log_return_lag{lag}"] = merged["log_return"].shift(lag)

    for horizon in range(MAX_HORIZON + 1):
        cumulative_return = sum(merged["log_return"].shift(-step) for step in range(horizon + 1))
        merged[f"cum_return_h{horizon}"] = cumulative_return

    shock_day_mask = merged[SHOCK_NAMES].ne(0).any(axis=1)
    merged["is_pre_qe"] = merged["date"] < QE_START_DATE
    merged["is_post_qe"] = merged["date"] >= QE_START_DATE
    merged["is_pre_super_thursday"] = merged["date"] < SUPER_THURSDAY_START_DATE
    merged["is_post_super_thursday"] = merged["date"] >= SUPER_THURSDAY_START_DATE

    print(f"\nMerged diagnostics for {get_index_display_name(merged['index'].iloc[0])}")
    print(f"  - Total trading-day observations: {len(merged):,}")
    print(f"  - Non-missing return observations: {int(merged['log_return'].notna().sum()):,}")
    print(f"  - Trading days with any shock: {int(shock_day_mask.sum()):,}")

    return merged


def save_merged_dataset(df: pd.DataFrame, index_name: str) -> None:
    output_path = TABLES_DIR / f"merged_{index_name}_v2.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved merged dataset: {output_path}")


def print_sample_diagnostics(df: pd.DataFrame, index_name: str, sample_name: str) -> None:
    print(f"\nSample diagnostics: {get_index_display_name(index_name)} | {get_sample_display_name(sample_name)}")
    print(f"  - Observations: {len(df):,}")
    print(f"  - Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    if "in_estimation_sample" in df.columns:
        estimation_df = df[df["in_estimation_sample"]].copy()
        print(f"  - Estimation-sample observations: {len(estimation_df):,}")
        print(f"  - Estimation start: {estimation_df['date'].min().date()}")
    print(f"  - Return observations: {int(df['log_return'].notna().sum()):,}")
    for shock in SHOCK_NAMES:
        nonzero_days = int(df[shock].ne(0).sum())
        print(f"  - Non-zero {shock} days: {nonzero_days:,}")


def split_samples(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    samples = {
        "full": df.copy(),
        "pre_2009": df[df["is_pre_qe"]].copy(),
        "post_2009": df[df["is_post_qe"]].copy(),
        "pre_super_thursday": df[df["is_pre_super_thursday"]].copy(),
        "post_super_thursday": df[df["is_post_super_thursday"]].copy(),
    }
    return samples


def get_qe_analysis_sample(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_post_qe"]].copy()


def replace_qe_rows(results_df: pd.DataFrame, qe_results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df

    non_qe_results = results_df[results_df["shock"] != "qe"].copy()
    qe_only_results = qe_results_df[qe_results_df["shock"] == "qe"].copy() if not qe_results_df.empty else pd.DataFrame()
    if qe_only_results.empty:
        return non_qe_results.reset_index(drop=True)

    combined = pd.concat([non_qe_results, qe_only_results], ignore_index=True)
    return combined.sort_values(["index", "sample", "spec", "shock", "horizon"]).reset_index(drop=True)


def replace_qe_magnitude_rows(magnitude_df: pd.DataFrame, qe_magnitude_df: pd.DataFrame) -> pd.DataFrame:
    if magnitude_df.empty:
        return magnitude_df

    non_qe_magnitude = magnitude_df[magnitude_df["shock"] != "qe"].copy()
    qe_only_magnitude = qe_magnitude_df[qe_magnitude_df["shock"] == "qe"].copy() if not qe_magnitude_df.empty else pd.DataFrame()
    if qe_only_magnitude.empty:
        return non_qe_magnitude.reset_index(drop=True)

    combined = pd.concat([non_qe_magnitude, qe_only_magnitude], ignore_index=True)
    return combined.sort_values(["index", "sample", "spec", "shock", "horizon"]).reset_index(drop=True)


def get_active_shocks(df: pd.DataFrame, spec: str, single_shock: str | None = None) -> list[str]:
    if spec == "single":
        if single_shock is None:
            raise ValueError("single_shock must be provided for single-shock estimation.")
        candidate_shocks = [single_shock]
    else:
        candidate_shocks = SHOCK_NAMES

    active_shocks = []
    for shock in candidate_shocks:
        tightening_nonzero = int(df[f"{shock}_tightening"].ne(0).sum())
        easing_nonzero = int(df[f"{shock}_easing"].ne(0).sum())
        total_abs = float(df[shock].abs().sum())

        if total_abs == 0 or (tightening_nonzero == 0 and easing_nonzero == 0):
            print(
                f"  - Skipping {shock} in this sample because there is effectively no variation."
            )
            continue

        active_shocks.append(shock)

    return active_shocks


def run_lp_regressions(
    df: pd.DataFrame,
    index_name: str,
    sample_name: str,
    spec: str,
    single_shock: str | None = None,
) -> pd.DataFrame:
    active_shocks = get_active_shocks(df, spec=spec, single_shock=single_shock)
    if not active_shocks:
        print(
            f"  - No estimable shocks for {get_index_display_name(index_name)} | "
            f"{get_sample_display_name(sample_name)} | {spec}."
        )
        return pd.DataFrame()

    lag_vars = [f"log_return_lag{lag}" for lag in range(1, RETURN_LAGS + 1)]
    shock_vars = []
    for shock in active_shocks:
        shock_vars.extend([f"{shock}_tightening", f"{shock}_easing"])

    results = []
    print(
        f"\nEstimating {spec} specification for {get_index_display_name(index_name)} | "
        f"{get_sample_display_name(sample_name)}"
    )
    print(f"  - Active shocks: {', '.join(active_shocks)}")

    for horizon in range(MAX_HORIZON + 1):
        y_var = f"cum_return_h{horizon}"
        regression_df = df[["date", "in_estimation_sample", y_var] + shock_vars + lag_vars].dropna().copy()
        regression_df = regression_df[regression_df["in_estimation_sample"]].copy()

        if regression_df.empty:
            print(f"  - Horizon {horizon}: no usable observations.")
            continue

        y = regression_df[y_var]
        x = sm.add_constant(regression_df[shock_vars + lag_vars], has_constant="add")
        model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": horizon + 1})

        for shock in active_shocks:
            tightening_var = f"{shock}_tightening"
            easing_var = f"{shock}_easing"

            beta_tightening = float(model.params[tightening_var])
            se_tightening = float(model.bse[tightening_var])
            beta_easing = float(model.params[easing_var])
            se_easing = float(model.bse[easing_var])
            pvalue_tightening = float(model.pvalues[tightening_var])
            pvalue_easing = float(model.pvalues[easing_var])

            restriction = np.zeros((1, len(model.params)))
            param_names = list(model.params.index)
            restriction[0, param_names.index(tightening_var)] = 1.0
            restriction[0, param_names.index(easing_var)] = -1.0
            asymmetry_test = model.t_test(restriction)

            results.append(
                {
                    "index": index_name,
                    "sample": sample_name,
                    "spec": spec,
                    "shock": shock,
                    "horizon": horizon,
                    "beta_tightening": beta_tightening,
                    "se_tightening": se_tightening,
                    "pvalue_tightening": pvalue_tightening,
                    "ci_lower_tightening": beta_tightening - CONF_Z * se_tightening,
                    "ci_upper_tightening": beta_tightening + CONF_Z * se_tightening,
                    "beta_easing": beta_easing,
                    "se_easing": se_easing,
                    "pvalue_easing": pvalue_easing,
                    "ci_lower_easing": beta_easing - CONF_Z * se_easing,
                    "ci_upper_easing": beta_easing + CONF_Z * se_easing,
                    "asymmetry_test_stat": float(np.asarray(asymmetry_test.tvalue).squeeze()),
                    "asymmetry_pvalue": float(np.asarray(asymmetry_test.pvalue).squeeze()),
                    "nobs": int(model.nobs),
                    "r_squared": float(model.rsquared),
                    "num_regressors": len(shock_vars) + len(lag_vars) + 1,
                }
            )

    return pd.DataFrame(results)


def build_magnitude_summary(results_df: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    shock_std_map = {
        shock: float(merged_df[shock].std(ddof=1))
        for shock in SHOCK_NAMES
    }

    selected = results_df[results_df["horizon"].isin(SELECTED_MAGNITUDE_HORIZONS)].copy()
    if selected.empty:
        return pd.DataFrame()

    selected["shock_std"] = selected["shock"].map(shock_std_map)

    selected["response_1sd_tightening"] = selected["beta_tightening"] * selected["shock_std"]
    selected["response_1sd_easing"] = selected["beta_easing"] * selected["shock_std"]
    selected["cum_effect_1sd_tightening"] = selected["response_1sd_tightening"]
    selected["cum_effect_1sd_easing"] = selected["response_1sd_easing"]

    columns = [
        "index",
        "sample",
        "spec",
        "shock",
        "horizon",
        "shock_std",
        "response_1sd_tightening",
        "response_1sd_easing",
        "cum_effect_1sd_tightening",
        "cum_effect_1sd_easing",
        "asymmetry_pvalue",
        "nobs",
    ]
    return selected[columns].sort_values(["index", "sample", "spec", "shock", "horizon"]).reset_index(drop=True)


def save_results_tables(results_df: pd.DataFrame, magnitude_df: pd.DataFrame, index_name: str, sample_name: str, spec: str) -> None:
    if not results_df.empty:
        results_path = TABLES_DIR / f"lp_results_{index_name}_{sample_name}_{spec}_v2.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Saved regression table: {results_path}")

    if not magnitude_df.empty:
        magnitude_path = TABLES_DIR / f"lp_magnitude_{index_name}_{sample_name}_{spec}_v2.csv"
        magnitude_df.to_csv(magnitude_path, index=False)
        print(f"Saved magnitude table: {magnitude_path}")


def plot_irf(results_df: pd.DataFrame, index_name: str, sample_name: str, spec: str, shock: str) -> None:
    subset = results_df[results_df["shock"] == shock].sort_values("horizon")
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    horizons = subset["horizon"]

    ax.plot(horizons, subset["beta_tightening"], color="tab:blue", linewidth=2, label="Tightening shock")
    ax.fill_between(horizons, subset["ci_lower_tightening"], subset["ci_upper_tightening"], color="tab:blue", alpha=0.2)

    ax.plot(horizons, subset["beta_easing"], color="tab:red", linewidth=2, label="Easing shock")
    ax.fill_between(horizons, subset["ci_lower_easing"], subset["ci_upper_easing"], color="tab:red", alpha=0.2)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(
        f"{get_index_display_name(index_name)} | {shock.upper()} | "
        f"{get_sample_display_name(sample_name)} | {spec}"
    )
    ax.set_xlabel("Horizon (days)")
    ax.set_ylabel("Cumulative log return response")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    output_path = FIGURES_DIR / f"irf_{index_name}_{sample_name}_{spec}_{shock}_v2.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def plot_index_comparison(all_results: pd.DataFrame, spec: str, sample_name: str, shock: str) -> None:
    subset = all_results[
        (all_results["spec"] == spec)
        & (all_results["sample"] == sample_name)
        & (all_results["shock"] == shock)
        & (all_results["index"].isin(["ftse250", "ftse100"]))
    ].copy()
    if subset.empty or subset["index"].nunique() < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    style_map = {"ftse250": "tab:blue", "ftse100": "tab:orange"}
    line_labels = {
        "ftse250": get_index_display_name("ftse250"),
        "ftse100": get_index_display_name("ftse100"),
    }

    for index_name in ["ftse250", "ftse100"]:
        index_subset = subset[subset["index"] == index_name].sort_values("horizon")
        if index_subset.empty:
            continue

        axes[0].plot(index_subset["horizon"], index_subset["beta_tightening"], color=style_map[index_name], linewidth=2, label=line_labels[index_name])
        axes[0].fill_between(index_subset["horizon"], index_subset["ci_lower_tightening"], index_subset["ci_upper_tightening"], color=style_map[index_name], alpha=0.15)

        axes[1].plot(index_subset["horizon"], index_subset["beta_easing"], color=style_map[index_name], linewidth=2, label=line_labels[index_name])
        axes[1].fill_between(index_subset["horizon"], index_subset["ci_lower_easing"], index_subset["ci_upper_easing"], color=style_map[index_name], alpha=0.15)

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Tightening shock")
    axes[1].set_title("Easing shock")
    axes[0].set_ylabel("Cumulative log return response")

    for ax in axes:
        ax.set_xlabel("Horizon (days)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)

    fig.suptitle(f"Index Comparison | {shock.upper()} | {sample_name} | {spec}")
    output_path = FIGURES_DIR / f"comparison_indices_{sample_name}_{spec}_{shock}_v2.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def plot_all_index_benchmark(all_results: pd.DataFrame, spec: str, sample_name: str, shock: str) -> None:
    subset = all_results[
        (all_results["spec"] == spec)
        & (all_results["sample"] == sample_name)
        & (all_results["shock"] == shock)
        & (all_results["index"].isin(["ftse250", "ftse100", "ftse_all_share"]))
    ].copy()
    if subset.empty or subset["index"].nunique() < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    style_map = {
        "ftse250": "tab:blue",
        "ftse100": "tab:orange",
        "ftse_all_share": "tab:gray",
    }

    for index_name in ["ftse250", "ftse100", "ftse_all_share"]:
        index_subset = subset[subset["index"] == index_name].sort_values("horizon")
        if index_subset.empty:
            continue

        label = get_index_display_name(index_name)
        axes[0].plot(index_subset["horizon"], index_subset["beta_tightening"], color=style_map[index_name], linewidth=2, label=label)
        axes[1].plot(index_subset["horizon"], index_subset["beta_easing"], color=style_map[index_name], linewidth=2, label=label)

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Tightening shock")
    axes[1].set_title("Easing shock")
    axes[0].set_ylabel("Cumulative log return response")

    for ax in axes:
        ax.set_xlabel("Horizon (days)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)

    fig.suptitle(f"All-Index Benchmark | {shock.upper()} | {sample_name} | {spec}")
    output_path = FIGURES_DIR / f"comparison_all_indices_{sample_name}_{spec}_{shock}_v2.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def plot_subsample_comparison(
    all_results: pd.DataFrame,
    index_name: str,
    spec: str,
    shock: str,
    sample_names: tuple[str, str] = ("pre_2009", "post_2009"),
    line_labels: dict[str, str] | None = None,
    output_suffix: str = "",
) -> None:
    if line_labels is None:
        line_labels = {
            "pre_2009": "Pre-2009 (before 2009-03-05)",
            "post_2009": "Post-2009 (on/after 2009-03-05)",
        }
    if shock == "qe" and sample_names == ("pre_2009", "post_2009"):
        return

    subset = all_results[
        (all_results["index"] == index_name)
        & (all_results["spec"] == spec)
        & (all_results["sample"].isin(sample_names))
        & (all_results["shock"] == shock)
    ].copy()
    if subset.empty or subset["sample"].nunique() < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    style_map = {
        "full": "tab:blue",
        "pre_2009": "tab:green",
        "post_2009": "tab:purple",
        "pre_super_thursday": "tab:green",
        "post_super_thursday": "tab:purple",
    }

    for sample_name in sample_names:
        sample_subset = subset[subset["sample"] == sample_name].sort_values("horizon")
        if sample_subset.empty:
            continue

        axes[0].plot(sample_subset["horizon"], sample_subset["beta_tightening"], color=style_map[sample_name], linewidth=2, label=line_labels[sample_name])
        axes[0].fill_between(sample_subset["horizon"], sample_subset["ci_lower_tightening"], sample_subset["ci_upper_tightening"], color=style_map[sample_name], alpha=0.15)

        axes[1].plot(sample_subset["horizon"], sample_subset["beta_easing"], color=style_map[sample_name], linewidth=2, label=line_labels[sample_name])
        axes[1].fill_between(sample_subset["horizon"], sample_subset["ci_lower_easing"], sample_subset["ci_upper_easing"], color=style_map[sample_name], alpha=0.15)

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Tightening shock")
    axes[1].set_title("Easing shock")
    axes[0].set_ylabel("Cumulative log return response")

    for ax in axes:
        ax.set_xlabel("Horizon (days)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)

    fig.suptitle(f"Sub-sample Comparison | {get_index_display_name(index_name)} | {shock.upper()} | {spec}")
    output_path = FIGURES_DIR / f"comparison_subsamples{output_suffix}_{index_name}_{spec}_{shock}_v2.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def save_combined_tables(all_results: pd.DataFrame, all_magnitudes: pd.DataFrame) -> None:
    if not all_results.empty:
        results_path = TABLES_DIR / "lp_results_all_v2.csv"
        all_results.to_csv(results_path, index=False)
        print(f"Saved combined regression table: {results_path}")

    if not all_magnitudes.empty:
        magnitude_path = TABLES_DIR / "lp_magnitude_all_v2.csv"
        all_magnitudes.to_csv(magnitude_path, index=False)
        print(f"Saved combined magnitude table: {magnitude_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local projections for UK equity responses to monetary shocks.")
    parser.add_argument(
        "--indices",
        nargs="+",
        default=["ftse250", "ftse100", "ftse_all_share"],
        choices=list(INDEX_DISPLAY_NAMES.keys()),
        help="Indices to estimate. Example: --indices ftse250 or --indices ftse250 ftse100 ftse_all_share",
    )
    parser.add_argument(
        "--specs",
        nargs="+",
        default=["joint", "single"],
        choices=["joint", "single"],
        help="Specifications to estimate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    print("v2 local projection setup")
    print(f"  - Indices: {', '.join(args.indices)}")
    print(f"  - Specifications: {', '.join(args.specs)}")
    print(f"  - Equity file: {EQUITY_FILE.name}")
    print(f"  - Maximum horizon: {MAX_HORIZON}")
    print(f"  - Return lags: {RETURN_LAGS}")
    print(f"  - QE split date: {QE_START_DATE.date()}")
    print(f"  - Super Thursday split date: {SUPER_THURSDAY_START_DATE.date()}")
    print("  - Missing shocks on trading days are set to zero after merging.")
    print("  - Joint specification includes target, path, and qe tightening and easing magnitudes together.")
    print("  - Single specification estimates one shock type at a time for robustness.")

    shocks = load_shock_data(SHOCK_FILE)

    all_results_list = []
    all_magnitude_list = []

    for index_name in args.indices:
        equity = load_equity_index_from_consolidated(EQUITY_FILE, index_name=index_name)
        merged = merge_equity_and_shocks(equity, shocks)
        save_merged_dataset(merged, index_name)

        samples = split_samples(merged)
        for sample_name, sample_df in samples.items():
            if sample_df.empty:
                print(f"\nSkipping empty sample: {get_index_display_name(index_name)} | {sample_name}")
                continue

            print_sample_diagnostics(sample_df, index_name, sample_name)

            if "joint" in args.specs:
                joint_results = run_lp_regressions(
                    sample_df,
                    index_name=index_name,
                    sample_name=sample_name,
                    spec="joint",
                )
                qe_sample_df = get_qe_analysis_sample(sample_df)
                qe_joint_results = (
                    run_lp_regressions(
                        qe_sample_df,
                        index_name=index_name,
                        sample_name=sample_name,
                        spec="joint",
                    )
                    if not qe_sample_df.empty
                    else pd.DataFrame()
                )
                joint_results = replace_qe_rows(joint_results, qe_joint_results)
                joint_magnitude = build_magnitude_summary(joint_results, sample_df)
                qe_joint_magnitude = build_magnitude_summary(qe_joint_results, qe_sample_df) if not qe_joint_results.empty else pd.DataFrame()
                joint_magnitude = replace_qe_magnitude_rows(joint_magnitude, qe_joint_magnitude)
                save_results_tables(joint_results, joint_magnitude, index_name, sample_name, "joint")

                if not joint_results.empty:
                    all_results_list.append(joint_results)
                if not joint_magnitude.empty:
                    all_magnitude_list.append(joint_magnitude)

                for shock in SHOCK_NAMES:
                    plot_irf(joint_results, index_name, sample_name, "joint", shock)

            if "single" in args.specs:
                for shock in SHOCK_NAMES:
                    single_results = run_lp_regressions(
                        sample_df,
                        index_name=index_name,
                        sample_name=sample_name,
                        spec="single",
                        single_shock=shock,
                    )
                    qe_sample_df = get_qe_analysis_sample(sample_df)
                    qe_single_results = (
                        run_lp_regressions(
                            qe_sample_df,
                            index_name=index_name,
                            sample_name=sample_name,
                            spec="single",
                            single_shock=shock,
                        )
                        if (shock == "qe" and not qe_sample_df.empty)
                        else pd.DataFrame()
                    )
                    if shock == "qe":
                        single_results = replace_qe_rows(single_results, qe_single_results)
                    single_magnitude = build_magnitude_summary(single_results, sample_df)
                    qe_single_magnitude = build_magnitude_summary(qe_single_results, qe_sample_df) if not qe_single_results.empty else pd.DataFrame()
                    if shock == "qe":
                        single_magnitude = replace_qe_magnitude_rows(single_magnitude, qe_single_magnitude)
                    save_results_tables(single_results, single_magnitude, index_name, sample_name, f"single_{shock}")

                    if not single_results.empty:
                        all_results_list.append(single_results)
                    if not single_magnitude.empty:
                        all_magnitude_list.append(single_magnitude)

                    plot_irf(single_results, index_name, sample_name, "single", shock)

    all_results = pd.concat(all_results_list, ignore_index=True) if all_results_list else pd.DataFrame()
    all_magnitudes = pd.concat(all_magnitude_list, ignore_index=True) if all_magnitude_list else pd.DataFrame()

    save_combined_tables(all_results, all_magnitudes)

    if not all_results.empty:
        for spec in sorted(all_results["spec"].unique()):
            for shock in SHOCK_NAMES:
                plot_index_comparison(all_results, spec=spec, sample_name="full", shock=shock)
                plot_all_index_benchmark(all_results, spec=spec, sample_name="full", shock=shock)

        for index_name in args.indices:
            for spec in sorted(all_results["spec"].unique()):
                for shock in SHOCK_NAMES:
                    plot_subsample_comparison(all_results, index_name=index_name, spec=spec, shock=shock)
                    plot_subsample_comparison(
                        all_results,
                        index_name=index_name,
                        spec=spec,
                        shock=shock,
                        sample_names=("pre_super_thursday", "post_super_thursday"),
                        line_labels={
                            "pre_super_thursday": "Pre-Super Thursday (before 2015-08-06)",
                            "post_super_thursday": "Post-Super Thursday (on/after 2015-08-06)",
                        },
                        output_suffix="_super_thursday",
                    )


if __name__ == "__main__":
    main()
