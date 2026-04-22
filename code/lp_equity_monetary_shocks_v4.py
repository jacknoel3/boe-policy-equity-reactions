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
OUTPUT_DIR = BASE_DIR / "output_v4"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

SHOCK_FILE = DATA_DIR / "measuring-monetary-policy-in-the-uk-the-ukmpesd.xlsx"
EQUITY_FILE = DATA_DIR / "uk_equity_indices_PX_LAST_19970101_20260312.csv"

TARGET_INDEX = "ftse250"
CONTROL_INDICES = ("ftse100", "ftse_all_share")
INDEX_DISPLAY_NAMES = {
    "ftse100": "FTSE 100",
    "ftse250": "FTSE 250",
    "ftse_all_share": "FTSE All-Share",
}

SHOCK_NAMES = ["target", "path", "qe"]
MAX_HORIZON = 20
MAX_RETURN_LAGS = 10
CONTROL_LAGS = 5
CONF_Z = 1.96
QE_START_DATE = "2009-03-05"
SUPER_THURSDAY_START_DATE = "2015-08-06"
COVID_START_DATE = "2020-03-01"
QT_START_DATE = "2022-01-01"

REGIME_DATE_MAP = {
    "post_qe": pd.Timestamp(QE_START_DATE),
    "post_super_thursday": pd.Timestamp(SUPER_THURSDAY_START_DATE),
    "covid_period": pd.Timestamp(COVID_START_DATE),
    "qt_period": pd.Timestamp(QT_START_DATE),
}
REGIME_DUMMIES = tuple(REGIME_DATE_MAP)
REGIME_DISPLAY_NAMES = {
    "baseline": "Baseline",
    "post_qe": "Post QE",
    "post_super_thursday": "Post Super Thursday",
    "covid_period": "COVID onward",
    "qt_period": "QT onward",
}

ROBUSTNESS_SPECS = (
    {"name": "baseline", "return_lags": 5, "controls": ()},
    {"name": "lag10", "return_lags": 10, "controls": ()},
    {"name": "ftse100_controls", "return_lags": 5, "controls": ("ftse100",)},
    {"name": "allshare_controls", "return_lags": 5, "controls": ("ftse_all_share",)},
    {"name": "dual_controls", "return_lags": 5, "controls": ("ftse100", "ftse_all_share")},
)


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def get_index_display_name(index_name: str) -> str:
    return INDEX_DISPLAY_NAMES.get(index_name, index_name.upper())


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
    return shocks


def load_equity_inputs() -> dict[str, pd.DataFrame]:
    equity_inputs = {
        TARGET_INDEX: load_equity_index_from_consolidated(EQUITY_FILE, index_name=TARGET_INDEX),
    }
    for control_index in CONTROL_INDICES:
        control_df = load_equity_index_from_consolidated(EQUITY_FILE, index_name=control_index)
        equity_inputs[control_index] = control_df[["date", "log_return"]].rename(
            columns={"log_return": f"{control_index}_return"}
        )
    return equity_inputs


def add_cumulative_returns(df: pd.DataFrame) -> pd.DataFrame:
    for horizon in range(MAX_HORIZON + 1):
        cumulative_return = sum(df["log_return"].shift(-step) for step in range(horizon + 1))
        df[f"cum_return_h{horizon}"] = cumulative_return
    return df


def keep_lag_buffer_before_shock_sample(target_equity: pd.DataFrame, shocks: pd.DataFrame) -> pd.DataFrame:
    shock_sample_start = shocks["date"].min()
    buffer_start = shock_sample_start - pd.tseries.offsets.BDay(MAX_RETURN_LAGS)
    buffered = target_equity[target_equity["date"] >= buffer_start].copy()
    buffered["shock_sample_start"] = shock_sample_start
    buffered["in_estimation_sample"] = buffered["date"] >= shock_sample_start

    print(f"Keeping lag buffer from: {buffer_start.date()}")
    print(f"  - Estimation sample starts: {shock_sample_start.date()}")
    print(f"  - Trading-day observations retained: {len(buffered):,}")
    return buffered


def build_merged_dataset(target_equity: pd.DataFrame, control_data: dict[str, pd.DataFrame], shocks: pd.DataFrame) -> pd.DataFrame:
    target_equity = keep_lag_buffer_before_shock_sample(target_equity, shocks)
    merged = target_equity.merge(shocks, on="date", how="left")
    merged[SHOCK_NAMES] = merged[SHOCK_NAMES].fillna(0.0)

    for control_index, control_df in control_data.items():
        merged = merged.merge(control_df, on="date", how="left")

    for shock in SHOCK_NAMES:
        merged[f"{shock}_pos"] = np.maximum(merged[shock], 0.0)
        merged[f"{shock}_neg"] = np.minimum(merged[shock], 0.0)

    for lag in range(1, MAX_RETURN_LAGS + 1):
        merged[f"log_return_lag{lag}"] = merged["log_return"].shift(lag)

    for control_index in CONTROL_INDICES:
        control_return_col = f"{control_index}_return"
        for lag in range(0, CONTROL_LAGS + 1):
            merged[f"{control_return_col}_lag{lag}"] = merged[control_return_col].shift(lag)

    for regime_dummy, regime_start in REGIME_DATE_MAP.items():
        merged[regime_dummy] = (merged["date"] >= regime_start).astype(int)

    merged["is_pre_qe"] = merged["date"] < REGIME_DATE_MAP["post_qe"]
    merged["is_post_qe"] = merged["post_qe"].astype(bool)
    merged["is_pre_super_thursday"] = merged["date"] < REGIME_DATE_MAP["post_super_thursday"]
    merged["is_post_super_thursday"] = merged["post_super_thursday"].astype(bool)

    return add_cumulative_returns(merged)


def split_samples(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "full": df.copy(),
        "pre_qe": df[df["is_pre_qe"]].copy(),
        "post_qe": df[df["is_post_qe"]].copy(),
        "pre_super_thursday": df[df["is_pre_super_thursday"]].copy(),
        "post_super_thursday": df[df["is_post_super_thursday"]].copy(),
        "covid_onward": df[df["covid_period"].astype(bool)].copy(),
    }


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
    return combined.sort_values(["sample", "robustness_spec", "shock", "horizon"]).reset_index(drop=True)


def replace_qe_magnitude_rows(magnitude_df: pd.DataFrame, qe_magnitude_df: pd.DataFrame) -> pd.DataFrame:
    if magnitude_df.empty:
        return magnitude_df

    non_qe_magnitude = magnitude_df[magnitude_df["shock"] != "qe"].copy()
    qe_only_magnitude = qe_magnitude_df[qe_magnitude_df["shock"] == "qe"].copy() if not qe_magnitude_df.empty else pd.DataFrame()
    if qe_only_magnitude.empty:
        return non_qe_magnitude.reset_index(drop=True)

    combined = pd.concat([non_qe_magnitude, qe_only_magnitude], ignore_index=True)
    return combined.sort_values(["sample", "robustness_spec", "shock", "horizon"]).reset_index(drop=True)


def get_active_shocks(df: pd.DataFrame) -> list[str]:
    active_shocks = []
    for shock in SHOCK_NAMES:
        if float(df[shock].abs().sum()) == 0.0:
            continue
        if int(df[f"{shock}_pos"].ne(0).sum()) == 0 and int(df[f"{shock}_neg"].ne(0).sum()) == 0:
            continue
        active_shocks.append(shock)
    return active_shocks


def get_control_columns(spec: dict[str, object]) -> list[str]:
    control_columns: list[str] = []
    for lag in range(1, int(spec["return_lags"]) + 1):
        control_columns.append(f"log_return_lag{lag}")

    for control_index in spec["controls"]:
        control_return_col = f"{control_index}_return"
        for lag in range(0, CONTROL_LAGS + 1):
            control_columns.append(f"{control_return_col}_lag{lag}")

    return control_columns


def run_lp_regressions(df: pd.DataFrame, sample_name: str, spec: dict[str, object]) -> pd.DataFrame:
    active_shocks = get_active_shocks(df)
    if not active_shocks:
        return pd.DataFrame()

    control_columns = get_control_columns(spec)
    shock_columns: list[str] = []
    for shock in active_shocks:
        shock_columns.extend([f"{shock}_pos", f"{shock}_neg"])

    results: list[dict[str, object]] = []
    for horizon in range(MAX_HORIZON + 1):
        y_var = f"cum_return_h{horizon}"
        regression_df = df[["date", "in_estimation_sample", y_var] + shock_columns + control_columns].dropna().copy()
        regression_df = regression_df[regression_df["in_estimation_sample"]].copy()
        if regression_df.empty:
            continue

        y = regression_df[y_var]
        x = sm.add_constant(regression_df[shock_columns + control_columns], has_constant="add")
        model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": horizon + 1})
        param_names = list(model.params.index)

        for shock in active_shocks:
            pos_var = f"{shock}_pos"
            neg_var = f"{shock}_neg"

            restriction = np.zeros((1, len(param_names)))
            restriction[0, param_names.index(pos_var)] = 1.0
            restriction[0, param_names.index(neg_var)] = -1.0
            asymmetry_test = model.t_test(restriction)

            beta_pos = float(model.params[pos_var])
            se_pos = float(model.bse[pos_var])
            beta_neg = float(model.params[neg_var])
            se_neg = float(model.bse[neg_var])

            results.append(
                {
                    "index": TARGET_INDEX,
                    "sample": sample_name,
                    "robustness_spec": str(spec["name"]),
                    "shock": shock,
                    "horizon": horizon,
                    "beta_pos": beta_pos,
                    "se_pos": se_pos,
                    "pvalue_pos": float(model.pvalues[pos_var]),
                    "ci_lower_pos": beta_pos - CONF_Z * se_pos,
                    "ci_upper_pos": beta_pos + CONF_Z * se_pos,
                    "beta_neg": beta_neg,
                    "se_neg": se_neg,
                    "pvalue_neg": float(model.pvalues[neg_var]),
                    "ci_lower_neg": beta_neg - CONF_Z * se_neg,
                    "ci_upper_neg": beta_neg + CONF_Z * se_neg,
                    "asymmetry_test_stat": float(np.asarray(asymmetry_test.tvalue).squeeze()),
                    "asymmetry_pvalue": float(np.asarray(asymmetry_test.pvalue).squeeze()),
                    "nobs": int(model.nobs),
                    "r_squared": float(model.rsquared),
                    "return_lags": int(spec["return_lags"]),
                    "control_set": "+".join(spec["controls"]) if spec["controls"] else "own_lags_only",
                }
            )

    return pd.DataFrame(results)


def get_regime_interaction_columns(df: pd.DataFrame, shock: str) -> list[str]:
    interaction_columns: list[str] = []
    for regime_dummy in REGIME_DUMMIES:
        if regime_dummy not in df.columns or df[regime_dummy].nunique(dropna=True) < 2:
            continue

        for shock_part in (f"{shock}_pos", f"{shock}_neg"):
            interaction_col = f"{shock_part}_x_{regime_dummy}"
            df[interaction_col] = df[shock_part] * df[regime_dummy]
            if float(df[interaction_col].abs().sum()) == 0.0:
                continue
            interaction_columns.append(interaction_col)

    return interaction_columns


def estimate_linear_combination(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    terms: dict[str, float],
) -> tuple[float, float, float, float, float]:
    param_names = list(model.params.index)
    restriction = np.zeros((1, len(param_names)))
    for term, weight in terms.items():
        if term in param_names:
            restriction[0, param_names.index(term)] = weight

    test_result = model.t_test(restriction)
    beta = float(np.asarray(test_result.effect).squeeze())
    se = float(np.asarray(test_result.sd).squeeze())
    pvalue = float(np.asarray(test_result.pvalue).squeeze())
    return beta, se, pvalue, beta - CONF_Z * se, beta + CONF_Z * se


def append_regime_response_row(
    results: list[dict[str, object]],
    model: sm.regression.linear_model.RegressionResultsWrapper,
    *,
    sample_name: str,
    spec: dict[str, object],
    shock: str,
    horizon: int,
    regime: str,
    pos_terms: dict[str, float],
    neg_terms: dict[str, float],
) -> None:
    beta_pos, se_pos, pvalue_pos, ci_lower_pos, ci_upper_pos = estimate_linear_combination(model, pos_terms)
    beta_neg, se_neg, pvalue_neg, ci_lower_neg, ci_upper_neg = estimate_linear_combination(model, neg_terms)

    results.append(
        {
            "index": TARGET_INDEX,
            "sample": sample_name,
            "robustness_spec": str(spec["name"]),
            "v4_specification": "regime_interactions",
            "shock": shock,
            "regime": regime,
            "regime_label": REGIME_DISPLAY_NAMES.get(regime, regime),
            "horizon": horizon,
            "beta_pos": beta_pos,
            "se_pos": se_pos,
            "pvalue_pos": pvalue_pos,
            "ci_lower_pos": ci_lower_pos,
            "ci_upper_pos": ci_upper_pos,
            "beta_neg": beta_neg,
            "se_neg": se_neg,
            "pvalue_neg": pvalue_neg,
            "ci_lower_neg": ci_lower_neg,
            "ci_upper_neg": ci_upper_neg,
            "nobs": int(model.nobs),
            "r_squared": float(model.rsquared),
            "return_lags": int(spec["return_lags"]),
            "control_set": "+".join(spec["controls"]) if spec["controls"] else "own_lags_only",
        }
    )


def run_regime_interaction_lp_regressions(df: pd.DataFrame, sample_name: str, spec: dict[str, object]) -> pd.DataFrame:
    control_columns = get_control_columns(spec)
    results: list[dict[str, object]] = []

    for shock in get_active_shocks(df):
        shock_df = get_qe_analysis_sample(df) if shock == "qe" else df.copy()
        if shock_df.empty or shock not in get_active_shocks(shock_df):
            continue

        base_shock_columns = [f"{shock}_pos", f"{shock}_neg"]
        interaction_columns = get_regime_interaction_columns(shock_df, shock)
        model_columns = base_shock_columns + interaction_columns + control_columns

        for horizon in range(MAX_HORIZON + 1):
            y_var = f"cum_return_h{horizon}"
            regression_df = shock_df[["date", "in_estimation_sample", y_var] + model_columns].dropna().copy()
            regression_df = regression_df[regression_df["in_estimation_sample"]].copy()
            if regression_df.empty:
                continue

            y = regression_df[y_var]
            x = sm.add_constant(regression_df[model_columns], has_constant="add")
            model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": horizon + 1})
            param_names = set(model.params.index)

            pos_var = f"{shock}_pos"
            neg_var = f"{shock}_neg"
            append_regime_response_row(
                results,
                model,
                sample_name=sample_name,
                spec=spec,
                shock=shock,
                horizon=horizon,
                regime="baseline",
                pos_terms={pos_var: 1.0},
                neg_terms={neg_var: 1.0},
            )

            for regime_dummy in REGIME_DUMMIES:
                pos_interaction = f"{pos_var}_x_{regime_dummy}"
                neg_interaction = f"{neg_var}_x_{regime_dummy}"
                if pos_interaction not in param_names or neg_interaction not in param_names:
                    continue

                append_regime_response_row(
                    results,
                    model,
                    sample_name=sample_name,
                    spec=spec,
                    shock=shock,
                    horizon=horizon,
                    regime=regime_dummy,
                    pos_terms={pos_var: 1.0, pos_interaction: 1.0},
                    neg_terms={neg_var: 1.0, neg_interaction: 1.0},
                )

    return pd.DataFrame(results)


def build_magnitude_summary(results_df: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    shock_std_map = {shock: float(merged_df[shock].std(ddof=1)) for shock in SHOCK_NAMES}
    selected = results_df[results_df["horizon"].isin([0, 5, 10, 20])].copy()
    if selected.empty:
        return pd.DataFrame()

    selected["shock_std"] = selected["shock"].map(shock_std_map)
    selected["response_1sd_tightening"] = selected["beta_pos"] * selected["shock_std"]
    selected["response_1sd_easing"] = selected["beta_neg"] * (-selected["shock_std"])

    columns = [
        "index",
        "sample",
        "robustness_spec",
        "shock",
        "horizon",
        "control_set",
        "return_lags",
        "shock_std",
        "response_1sd_tightening",
        "response_1sd_easing",
        "asymmetry_pvalue",
        "nobs",
    ]
    return selected[columns].sort_values(["sample", "robustness_spec", "shock", "horizon"]).reset_index(drop=True)


def save_tables(
    merged_df: pd.DataFrame,
    all_results: pd.DataFrame,
    all_magnitudes: pd.DataFrame,
    regime_results: pd.DataFrame,
) -> None:
    merged_path = TABLES_DIR / "merged_ftse250_v4.csv"
    merged_df.to_csv(merged_path, index=False)

    results_path = TABLES_DIR / "lp_results_ftse250_robustness_v4.csv"
    all_results.to_csv(results_path, index=False)

    magnitude_path = TABLES_DIR / "lp_magnitude_ftse250_robustness_v4.csv"
    all_magnitudes.to_csv(magnitude_path, index=False)

    regime_path = TABLES_DIR / "lp_regime_interactions_ftse250_v4.csv"
    regime_results.to_csv(regime_path, index=False)

    print(f"Saved merged dataset: {merged_path}")
    print(f"Saved regression table: {results_path}")
    print(f"Saved magnitude table: {magnitude_path}")
    print(f"Saved regime interaction table: {regime_path}")


def plot_robustness_comparison(all_results: pd.DataFrame, shock: str, sample_name: str = "full") -> None:
    subset = all_results[
        (all_results["shock"] == shock)
        & (all_results["sample"] == sample_name)
    ].copy()
    if subset.empty:
        return

    spec_order = [spec["name"] for spec in ROBUSTNESS_SPECS if spec["name"] in subset["robustness_spec"].unique()]
    color_map = {
        "baseline": "tab:blue",
        "lag10": "tab:orange",
        "ftse100_controls": "tab:green",
        "allshare_controls": "tab:red",
        "dual_controls": "tab:purple",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for spec_name in spec_order:
        spec_subset = subset[subset["robustness_spec"] == spec_name].sort_values("horizon")
        if spec_subset.empty:
            continue

        color = color_map.get(spec_name, None)
        axes[0].plot(spec_subset["horizon"], spec_subset["beta_pos"], color=color, linewidth=2, label=spec_name)
        axes[0].fill_between(
            spec_subset["horizon"],
            spec_subset["ci_lower_pos"],
            spec_subset["ci_upper_pos"],
            color=color,
            alpha=0.12,
        )
        axes[1].plot(spec_subset["horizon"], spec_subset["beta_neg"], color=color, linewidth=2, label=spec_name)
        axes[1].fill_between(
            spec_subset["horizon"],
            spec_subset["ci_lower_neg"],
            spec_subset["ci_upper_neg"],
            color=color,
            alpha=0.12,
        )

    axes[0].set_title("Tightening")
    axes[1].set_title("Easing")
    axes[0].set_ylabel("Cumulative log return response")
    for ax in axes:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Horizon (days)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)

    fig.suptitle(f"FTSE 250 Robustness | {shock.upper()} | {sample_name}")
    output_path = FIGURES_DIR / f"robustness_{shock}_{sample_name}_v4.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def plot_subsample_comparison(
    all_results: pd.DataFrame,
    shock: str,
    robustness_spec: str = "baseline",
    sample_order: list[str] | None = None,
    title_suffix: str = "",
    output_suffix: str = "",
) -> None:
    subset = all_results[
        (all_results["shock"] == shock)
        & (all_results["robustness_spec"] == robustness_spec)
    ].copy()
    if subset.empty:
        return

    if sample_order is None:
        sample_order = ["full", "pre_qe", "post_qe", "covid_onward"]
        if shock == "qe":
            sample_order = ["full", "covid_onward"]
    available_samples = [sample for sample in sample_order if sample in subset["sample"].unique()]
    if len(available_samples) < 2:
        return

    color_map = {
        "full": "tab:blue",
        "pre_qe": "tab:green",
        "post_qe": "tab:orange",
        "pre_super_thursday": "tab:green",
        "post_super_thursday": "tab:orange",
        "covid_onward": "tab:red",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for sample_name in available_samples:
        sample_subset = subset[subset["sample"] == sample_name].sort_values("horizon")
        color = color_map.get(sample_name, None)
        axes[0].plot(sample_subset["horizon"], sample_subset["beta_pos"], color=color, linewidth=2, label=sample_name)
        axes[0].fill_between(
            sample_subset["horizon"],
            sample_subset["ci_lower_pos"],
            sample_subset["ci_upper_pos"],
            color=color,
            alpha=0.12,
        )
        axes[1].plot(sample_subset["horizon"], sample_subset["beta_neg"], color=color, linewidth=2, label=sample_name)
        axes[1].fill_between(
            sample_subset["horizon"],
            sample_subset["ci_lower_neg"],
            sample_subset["ci_upper_neg"],
            color=color,
            alpha=0.12,
        )

    axes[0].set_title("Tightening")
    axes[1].set_title("Easing")
    axes[0].set_ylabel("Cumulative log return response")
    for ax in axes:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Horizon (days)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)

    fig.suptitle(f"FTSE 250 Subsamples{title_suffix} | {shock.upper()} | {robustness_spec}")
    output_path = FIGURES_DIR / f"subsamples{output_suffix}_{shock}_{robustness_spec}_v4.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def plot_regime_interaction_irfs(
    regime_results: pd.DataFrame,
    shock: str,
    robustness_spec: str = "baseline",
    sample_name: str = "full",
) -> None:
    subset = regime_results[
        (regime_results["shock"] == shock)
        & (regime_results["robustness_spec"] == robustness_spec)
        & (regime_results["sample"] == sample_name)
    ].copy()
    if subset.empty:
        return

    regime_order = ["baseline", *REGIME_DUMMIES]
    available_regimes = [regime for regime in regime_order if regime in subset["regime"].unique()]
    if len(available_regimes) < 2:
        return

    color_map = {
        "baseline": "tab:blue",
        "post_qe": "tab:orange",
        "post_super_thursday": "tab:green",
        "covid_period": "tab:red",
        "qt_period": "tab:purple",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for regime in available_regimes:
        regime_subset = subset[subset["regime"] == regime].sort_values("horizon")
        if regime_subset.empty:
            continue

        color = color_map.get(regime, None)
        label = REGIME_DISPLAY_NAMES.get(regime, regime)
        axes[0].plot(regime_subset["horizon"], regime_subset["beta_pos"], color=color, linewidth=2, label=label)
        axes[0].fill_between(
            regime_subset["horizon"],
            regime_subset["ci_lower_pos"],
            regime_subset["ci_upper_pos"],
            color=color,
            alpha=0.12,
        )
        axes[1].plot(regime_subset["horizon"], regime_subset["beta_neg"], color=color, linewidth=2, label=label)
        axes[1].fill_between(
            regime_subset["horizon"],
            regime_subset["ci_lower_neg"],
            regime_subset["ci_upper_neg"],
            color=color,
            alpha=0.12,
        )

    axes[0].set_title("Tightening")
    axes[1].set_title("Easing")
    axes[0].set_ylabel("Cumulative log return response")
    for ax in axes:
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Horizon (days)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)

    fig.suptitle(f"FTSE 250 Regime Interactions | {shock.upper()} | {robustness_spec}")
    output_path = FIGURES_DIR / f"regime_interactions_{shock}_{robustness_spec}_v4.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FTSE 250 local projections with v4 regime interactions.")
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["full", "pre_qe", "post_qe", "pre_super_thursday", "post_super_thursday", "covid_onward"],
        choices=["full", "pre_qe", "post_qe", "pre_super_thursday", "post_super_thursday", "covid_onward"],
        help="Subsamples to estimate.",
    )
    parser.add_argument(
        "--specs",
        nargs="+",
        default=[spec["name"] for spec in ROBUSTNESS_SPECS],
        choices=[spec["name"] for spec in ROBUSTNESS_SPECS],
        help="Robustness specifications to estimate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    print("v4 FTSE 250 local projection robustness and regime-interaction setup")
    print(f"  - Target index: {get_index_display_name(TARGET_INDEX)}")
    print(f"  - Samples: {', '.join(args.samples)}")
    print(f"  - Robustness specs: {', '.join(args.specs)}")
    print(f"  - Equity file: {EQUITY_FILE.name}")
    print(f"  - Shock file: {SHOCK_FILE.name}")
    print(f"  - Maximum horizon: {MAX_HORIZON}")
    print(f"  - Maximum own-return lags precomputed: {MAX_RETURN_LAGS}")
    print(f"  - Benchmark control lags precomputed: {CONTROL_LAGS}")
    print(f"  - QE regime date: {QE_START_DATE}")
    print(f"  - Super Thursday regime date: {SUPER_THURSDAY_START_DATE}")
    print(f"  - COVID regime date: {COVID_START_DATE}")
    print(f"  - QT regime date: {QT_START_DATE}")

    shocks = load_shock_data(SHOCK_FILE)
    equity_inputs = load_equity_inputs()
    merged = build_merged_dataset(
        target_equity=equity_inputs[TARGET_INDEX],
        control_data={name: equity_inputs[name] for name in CONTROL_INDICES},
        shocks=shocks,
    )

    sample_map = split_samples(merged)
    requested_specs = [spec for spec in ROBUSTNESS_SPECS if spec["name"] in args.specs]

    all_results_list: list[pd.DataFrame] = []
    all_magnitude_list: list[pd.DataFrame] = []
    regime_results_list: list[pd.DataFrame] = []

    for sample_name in args.samples:
        sample_df = sample_map[sample_name]
        if sample_df.empty:
            continue

        estimation_df = sample_df[sample_df["in_estimation_sample"]].copy()
        print(
            f"Estimating {sample_name}: {len(sample_df):,} rows kept "
            f"({len(estimation_df):,} estimation rows) from {sample_df['date'].min().date()} to {sample_df['date'].max().date()}"
        )

        for spec in requested_specs:
            results_df = run_lp_regressions(sample_df, sample_name=sample_name, spec=spec)
            if results_df.empty:
                continue

            qe_sample_df = get_qe_analysis_sample(sample_df)
            qe_results_df = run_lp_regressions(qe_sample_df, sample_name=sample_name, spec=spec) if not qe_sample_df.empty else pd.DataFrame()
            results_df = replace_qe_rows(results_df, qe_results_df)
            magnitudes_df = build_magnitude_summary(results_df, sample_df)
            qe_magnitudes_df = build_magnitude_summary(qe_results_df, qe_sample_df) if not qe_results_df.empty else pd.DataFrame()
            magnitudes_df = replace_qe_magnitude_rows(magnitudes_df, qe_magnitudes_df)
            all_results_list.append(results_df)
            if not magnitudes_df.empty:
                all_magnitude_list.append(magnitudes_df)

    print("Estimating v4 full-sample regime-interaction specifications")
    for spec in requested_specs:
        regime_results_df = run_regime_interaction_lp_regressions(merged, sample_name="full", spec=spec)
        if not regime_results_df.empty:
            regime_results_list.append(regime_results_df)

    all_results = pd.concat(all_results_list, ignore_index=True) if all_results_list else pd.DataFrame()
    all_magnitudes = pd.concat(all_magnitude_list, ignore_index=True) if all_magnitude_list else pd.DataFrame()
    regime_results = pd.concat(regime_results_list, ignore_index=True) if regime_results_list else pd.DataFrame()
    save_tables(merged, all_results, all_magnitudes, regime_results)

    if not all_results.empty:
        for shock in SHOCK_NAMES:
            plot_robustness_comparison(all_results, shock=shock, sample_name="full")
            plot_subsample_comparison(all_results, shock=shock, robustness_spec="baseline")
            plot_subsample_comparison(
                all_results,
                shock=shock,
                robustness_spec="baseline",
                sample_order=["full", "pre_super_thursday", "post_super_thursday", "covid_onward"],
                title_suffix=" | Super Thursday split",
                output_suffix="_super_thursday",
            )

    if not regime_results.empty:
        for shock in SHOCK_NAMES:
            plot_regime_interaction_irfs(regime_results, shock=shock, robustness_spec="baseline")


if __name__ == "__main__":
    main()
