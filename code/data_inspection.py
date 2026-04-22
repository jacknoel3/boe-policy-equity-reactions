"""Pre-analysis diagnostics for monetary-shock and UK equity source files.

The script audits source inputs without changing them. Outputs are written to
``data/diagnostics`` so the checks can be reviewed alongside the input data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import INDEX_ALIASES, _find_best_index_match, read_consolidated_equity_file


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DIAGNOSTICS_DIR = DATA_DIR / "diagnostics"

SHOCK_FILE = DATA_DIR / "measuring-monetary-policy-in-the-uk-the-ukmpesd.xlsx"
EQUITY_FILE = DATA_DIR / "uk_equity_indices_PX_LAST_19970101_20260312.csv"

SHOCK_SHEET = "factors"
SHOCK_NAMES = ["target", "path", "qe"]
INDEX_NAMES = ["ftse100", "ftse250", "ftse_all_share"]
OUTLIER_QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)
TOP_N_EXTREMES = 10


def ensure_output_dir() -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_columns(df: pd.DataFrame) -> dict[str, str]:
    return {str(col).strip().lower(): col for col in df.columns}


def load_shock_data() -> tuple[pd.DataFrame, list[str]]:
    workbook = pd.ExcelFile(SHOCK_FILE)
    shocks_raw = pd.read_excel(SHOCK_FILE, sheet_name=SHOCK_SHEET)
    normalized = normalize_columns(shocks_raw)

    datetime_col = normalized.get("datetime")
    if datetime_col is None:
        raise ValueError("Could not detect the shock datetime column.")

    missing = [shock for shock in SHOCK_NAMES if shock not in normalized]
    if missing:
        raise ValueError(f"Missing required shock columns in '{SHOCK_SHEET}': {missing}")

    shocks = shocks_raw[[datetime_col] + [normalized[shock] for shock in SHOCK_NAMES]].copy()
    shocks.columns = ["datetime"] + SHOCK_NAMES
    shocks["datetime"] = pd.to_datetime(shocks["datetime"], errors="coerce")
    shocks["date"] = shocks["datetime"].dt.normalize()
    for shock in SHOCK_NAMES:
        shocks[shock] = pd.to_numeric(shocks[shock], errors="coerce")

    return shocks, workbook.sheet_names


def load_equity_series() -> tuple[pd.DataFrame, dict[str, str], dict[str, dict[str, str]]]:
    raw, info = read_consolidated_equity_file(EQUITY_FILE)
    date_col = info["date_col"]
    security_col = info["security_col"] or None
    label_col = info["label_col"] or None
    price_col = info["price_col"]

    series_frames: list[pd.DataFrame] = []
    match_info_by_index: dict[str, dict[str, str]] = {}
    for index_name in INDEX_NAMES:
        if index_name not in INDEX_ALIASES:
            continue

        mask, match_info = _find_best_index_match(
            raw,
            index_name=index_name,
            security_col=security_col,
            label_col=label_col,
        )
        match_info_by_index[index_name] = match_info

        equity = raw.loc[mask, [date_col, price_col]].copy()
        equity = equity.rename(columns={date_col: "date", price_col: "price"})
        equity["date"] = pd.to_datetime(equity["date"], errors="coerce").dt.normalize()
        equity["price"] = pd.to_numeric(equity["price"], errors="coerce")
        equity = equity.dropna(subset=["date", "price"]).sort_values("date")
        equity = equity.drop_duplicates(subset=["date"]).reset_index(drop=True)
        equity["log_price"] = np.log(equity["price"])
        equity["log_return"] = equity["log_price"].diff()
        equity["index"] = index_name
        series_frames.append(equity)

    equity_series = pd.concat(series_frames, ignore_index=True) if series_frames else pd.DataFrame()
    return equity_series, info, match_info_by_index


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def summarize_numeric(series: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {
            "n": 0,
            "missing": int(series.isna().sum()),
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    quantiles = clean.quantile(OUTLIER_QUANTILES)
    return {
        "n": int(clean.size),
        "missing": int(series.isna().sum()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)),
        "min": float(clean.min()),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "median": float(quantiles.loc[0.5]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(clean.max()),
    }


def build_file_inventory(shocks: pd.DataFrame, equity_raw_info: dict[str, str], workbook_sheets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "file": SHOCK_FILE.name,
                "path": str(SHOCK_FILE.relative_to(BASE_DIR)),
                "size_bytes": SHOCK_FILE.stat().st_size,
                "role": "monetary policy shocks",
                "sheets_or_columns": "; ".join(workbook_sheets),
            },
            {
                "file": EQUITY_FILE.name,
                "path": str(EQUITY_FILE.relative_to(BASE_DIR)),
                "size_bytes": EQUITY_FILE.stat().st_size,
                "role": "consolidated equity prices",
                "sheets_or_columns": ", ".join(equity_raw_info.values()),
            },
        ]
    )


def build_shock_summary(shocks: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid_dates = shocks["date"].dropna()
    for shock in SHOCK_NAMES:
        summary = summarize_numeric(shocks[shock])
        summary.update(
            {
                "component": shock,
                "date_min": valid_dates.min().date() if not valid_dates.empty else None,
                "date_max": valid_dates.max().date() if not valid_dates.empty else None,
                "nonzero_count": int(shocks[shock].fillna(0.0).ne(0.0).sum()),
                "positive_count": int(shocks[shock].fillna(0.0).gt(0.0).sum()),
                "negative_count": int(shocks[shock].fillna(0.0).lt(0.0).sum()),
            }
        )
        rows.append(summary)

    return pd.DataFrame(rows)[
        [
            "component",
            "n",
            "missing",
            "date_min",
            "date_max",
            "nonzero_count",
            "positive_count",
            "negative_count",
            "mean",
            "std",
            "min",
            "p01",
            "p05",
            "median",
            "p95",
            "p99",
            "max",
        ]
    ]


def build_shock_extremes(shocks: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for shock in SHOCK_NAMES:
        clean = shocks[["date", shock]].dropna().copy()
        clean["component"] = shock
        clean["abs_value"] = clean[shock].abs()
        rows.append(clean.sort_values("abs_value", ascending=False).head(TOP_N_EXTREMES))

    extremes = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if extremes.empty:
        return extremes
    return extremes[["component", "date", "abs_value", *SHOCK_NAMES]].sort_values(["component", "abs_value"], ascending=[True, False])


def build_equity_summary(
    equity_series: pd.DataFrame,
    raw_info: dict[str, str],
    match_info_by_index: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index_name, group in equity_series.groupby("index"):
        returns = group["log_return"]
        summary = summarize_numeric(returns)
        match_info = match_info_by_index[index_name]
        rows.append(
            {
                "index": index_name,
                "match_source": match_info["match_source"],
                "match_value": match_info["match_value"],
                "date_col": raw_info["date_col"],
                "price_col": raw_info["price_col"],
                "observations": int(len(group)),
                "return_observations": int(returns.notna().sum()),
                "date_min": group["date"].min().date(),
                "date_max": group["date"].max().date(),
                "duplicate_dates_after_cleaning": int(group["date"].duplicated().sum()),
                "missing_prices_after_cleaning": int(group["price"].isna().sum()),
                "mean_return": summary["mean"],
                "std_return": summary["std"],
                "min_return": summary["min"],
                "p01_return": summary["p01"],
                "p05_return": summary["p05"],
                "median_return": summary["median"],
                "p95_return": summary["p95"],
                "p99_return": summary["p99"],
                "max_return": summary["max"],
            }
        )

    return pd.DataFrame(rows).sort_values("index").reset_index(drop=True)


def build_equity_extremes(equity_series: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for index_name, group in equity_series.dropna(subset=["log_return"]).groupby("index"):
        selected = group[["index", "date", "price", "log_return"]].copy()
        selected["abs_log_return"] = selected["log_return"].abs()
        rows.append(selected.sort_values("abs_log_return", ascending=False).head(TOP_N_EXTREMES))

    extremes = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if extremes.empty:
        return extremes
    return extremes.sort_values(["index", "abs_log_return"], ascending=[True, False]).reset_index(drop=True)


def build_merge_diagnostics(shocks: pd.DataFrame, equity_series: pd.DataFrame) -> pd.DataFrame:
    shock_dates = shocks.loc[shocks[SHOCK_NAMES].fillna(0.0).abs().sum(axis=1).gt(0.0), ["date", *SHOCK_NAMES]].copy()
    shock_dates = shock_dates.dropna(subset=["date"]).drop_duplicates(subset=["date"])

    rows: list[dict[str, object]] = []
    for index_name, group in equity_series.groupby("index"):
        trading_dates = set(group["date"])
        matched_mask = shock_dates["date"].isin(trading_dates)
        unmatched = shock_dates.loc[~matched_mask].copy()

        rows.append(
            {
                "index": index_name,
                "shock_event_dates": int(len(shock_dates)),
                "matched_trading_dates": int(matched_mask.sum()),
                "unmatched_event_dates": int((~matched_mask).sum()),
                "match_rate": float(matched_mask.mean()) if len(shock_dates) else np.nan,
                "first_unmatched_dates": ", ".join(unmatched["date"].dt.strftime("%Y-%m-%d").head(10).tolist()),
            }
        )

    return pd.DataFrame(rows).sort_values("index").reset_index(drop=True)


def build_report(
    *,
    file_inventory: pd.DataFrame,
    shock_summary: pd.DataFrame,
    shock_extremes: pd.DataFrame,
    equity_summary: pd.DataFrame,
    equity_extremes: pd.DataFrame,
    merge_diagnostics: pd.DataFrame,
) -> str:
    lines = [
        "# Data Diagnostics",
        "",
        "This report audits the source files before the local-projection analysis. Outlier tables flag notable observations; they do not imply deletion or winsorisation.",
        "",
        "## Files",
    ]

    for _, row in file_inventory.iterrows():
        lines.append(f"- `{row['path']}`: {row['role']} ({row['size_bytes']:,} bytes)")

    lines.extend(["", "## Monetary Shocks"])
    for _, row in shock_summary.iterrows():
        lines.append(
            f"- `{row['component']}`: {row['nonzero_count']:,} nonzero events, "
            f"range {row['date_min']} to {row['date_max']}, "
            f"std {row['std']:.6g}, min {row['min']:.6g}, max {row['max']:.6g}"
        )

    lines.extend(["", "## Equity Series"])
    for _, row in equity_summary.iterrows():
        lines.append(
            f"- `{row['index']}` matched on {row['match_source']}=`{row['match_value']}`: "
            f"{row['observations']:,} observations from {row['date_min']} to {row['date_max']}; "
            f"return min {row['min_return']:.4%}, max {row['max_return']:.4%}"
        )

    lines.extend(["", "## Event-Date Merge"])
    for _, row in merge_diagnostics.iterrows():
        lines.append(
            f"- `{row['index']}`: {row['matched_trading_dates']:,}/{row['shock_event_dates']:,} "
            f"shock event dates matched to equity trading dates ({pct(row['match_rate'])})."
        )
        if row["first_unmatched_dates"]:
            lines.append(f"  First unmatched dates: {row['first_unmatched_dates']}")

    lines.extend(
        [
            "",
            "## Output Files",
            "- `data_file_inventory.csv`",
            "- `shock_summary.csv`",
            "- `shock_extreme_observations.csv`",
            "- `equity_series_summary.csv`",
            "- `equity_return_extremes.csv`",
            "- `merge_diagnostics.csv`",
            "",
            "Largest observations are intentionally reported for transparency. In this application, large shocks or returns may be economically meaningful event observations rather than data errors.",
        ]
    )

    # Touch these frames so static checkers do not treat them as unused in the report contract.
    _ = shock_extremes, equity_extremes
    return "\n".join(lines) + "\n"


def save_outputs(outputs: dict[str, pd.DataFrame], report: str) -> None:
    for name, frame in outputs.items():
        path = DIAGNOSTICS_DIR / name
        frame.to_csv(path, index=False)
        print(f"Saved: {path}")

    report_path = DIAGNOSTICS_DIR / "diagnostic_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved: {report_path}")


def main() -> None:
    ensure_output_dir()

    shocks, workbook_sheets = load_shock_data()
    equity_series, equity_raw_info, match_info_by_index = load_equity_series()

    file_inventory = build_file_inventory(shocks, equity_raw_info, workbook_sheets)
    shock_summary = build_shock_summary(shocks)
    shock_extremes = build_shock_extremes(shocks)
    equity_summary = build_equity_summary(equity_series, equity_raw_info, match_info_by_index)
    equity_extremes = build_equity_extremes(equity_series)
    merge_diagnostics = build_merge_diagnostics(shocks, equity_series)

    outputs = {
        "data_file_inventory.csv": file_inventory,
        "shock_summary.csv": shock_summary,
        "shock_extreme_observations.csv": shock_extremes,
        "equity_series_summary.csv": equity_summary,
        "equity_return_extremes.csv": equity_extremes,
        "merge_diagnostics.csv": merge_diagnostics,
    }
    report = build_report(
        file_inventory=file_inventory,
        shock_summary=shock_summary,
        shock_extremes=shock_extremes,
        equity_summary=equity_summary,
        equity_extremes=equity_extremes,
        merge_diagnostics=merge_diagnostics,
    )
    save_outputs(outputs, report)

    print("\nPre-analysis data diagnostics complete.")
    print(f"Diagnostics folder: {DIAGNOSTICS_DIR}")


if __name__ == "__main__":
    main()
