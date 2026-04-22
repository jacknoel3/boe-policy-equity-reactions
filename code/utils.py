from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INDEX_ALIASES = {
    "ftse100": ["ftse 100", "ftse100", ".ftse", "ukx index", "ukx"],
    "ftse250": ["ftse 250", "ftse250", ".ftmc", "mcx index", "mcx"],
    "ftse_all_share": ["ftse all share", "ftse all-share", "ftseas", ".ftas", "asx index", "asx"],
}


def _normalize_text(value: object) -> str:
    text = str(value).strip().lower()
    for char in ["-", "_", ".", ","]:
        text = text.replace(char, " ")
    return " ".join(text.split())


def detect_equity_columns(equity_df: pd.DataFrame) -> tuple[str, str | None, str | None, str]:
    columns = list(equity_df.columns)
    normalized = {_normalize_text(col): col for col in columns}

    date_col = next((col for col in columns if "date" in _normalize_text(col)), None)
    if date_col is None:
        raise ValueError("Could not detect the equity date column in the consolidated file.")

    security_col = None
    for candidate in ["security", "ticker", "ric", "index"]:
        if candidate in normalized:
            security_col = normalized[candidate]
            break

    label_col = None
    for candidate in ["label", "name", "description", "index name"]:
        if candidate in normalized:
            label_col = normalized[candidate]
            break

    preferred_price_names = ["px last", "close", "price", "last", "adj close", "adj_close"]
    price_col = None
    for name in preferred_price_names:
        key = _normalize_text(name)
        if key in normalized:
            price_col = normalized[key]
            break

    if price_col is None:
        numeric_candidates = [col for col in columns if pd.api.types.is_numeric_dtype(equity_df[col])]
        numeric_candidates = [col for col in numeric_candidates if col != date_col]
        if not numeric_candidates:
            raise ValueError("Could not detect the equity price column in the consolidated file.")
        price_col = numeric_candidates[0]

    return date_col, security_col, label_col, price_col


def read_consolidated_equity_file(file_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Missing consolidated equity file: {file_path}")

    raw = pd.read_csv(file_path)
    date_col, security_col, label_col, price_col = detect_equity_columns(raw)

    info = {
        "date_col": date_col,
        "security_col": security_col or "",
        "label_col": label_col or "",
        "price_col": price_col,
    }

    return raw, info


def _find_best_index_match(raw: pd.DataFrame, index_name: str, security_col: str | None, label_col: str | None) -> tuple[pd.Series, dict[str, str]]:
    aliases = INDEX_ALIASES[index_name]
    candidates: list[tuple[int, pd.Series, dict[str, str]]] = []

    if label_col:
        label_values = raw[label_col].dropna().astype(str).unique().tolist()
        for value in label_values:
            normalized_value = _normalize_text(value)
            if normalized_value in aliases:
                mask = raw[label_col].astype(str).map(_normalize_text) == normalized_value
                candidates.append(
                    (
                        3,
                        mask,
                        {"match_source": label_col, "match_value": value},
                    )
                )
            elif any(alias in normalized_value for alias in aliases):
                mask = raw[label_col].astype(str).map(_normalize_text) == normalized_value
                candidates.append(
                    (
                        2,
                        mask,
                        {"match_source": label_col, "match_value": value},
                    )
                )

    if security_col:
        security_values = raw[security_col].dropna().astype(str).unique().tolist()
        for value in security_values:
            normalized_value = _normalize_text(value)
            if normalized_value in aliases:
                mask = raw[security_col].astype(str).map(_normalize_text) == normalized_value
                candidates.append(
                    (
                        2,
                        mask,
                        {"match_source": security_col, "match_value": value},
                    )
                )
            elif any(alias in normalized_value for alias in aliases):
                mask = raw[security_col].astype(str).map(_normalize_text) == normalized_value
                candidates.append(
                    (
                        1,
                        mask,
                        {"match_source": security_col, "match_value": value},
                    )
                )

    if not candidates:
        raise ValueError(
            f"Could not identify a column/value match for {index_name}. "
            "Please check the consolidated equity file naming."
        )

    candidates.sort(key=lambda item: (item[0], int(item[1].sum())), reverse=True)
    _, best_mask, best_info = candidates[0]
    return best_mask, best_info


def load_equity_index_from_consolidated(file_path: Path, index_name: str) -> pd.DataFrame:
    raw, info = read_consolidated_equity_file(file_path)
    date_col = info["date_col"]
    security_col = info["security_col"] or None
    label_col = info["label_col"] or None
    price_col = info["price_col"]

    print("\nConsolidated equity file diagnostics")
    print(f"  - File: {file_path.name}")
    print(f"  - Available columns: {list(raw.columns)}")
    print(f"  - Date column: {date_col}")
    print(f"  - Security column: {security_col}")
    print(f"  - Label column: {label_col}")
    print(f"  - Price column: {price_col}")

    if security_col:
        print(f"  - Unique security values: {sorted(raw[security_col].dropna().astype(str).unique().tolist())}")
    if label_col:
        print(f"  - Unique label values: {sorted(raw[label_col].dropna().astype(str).unique().tolist())}")

    mask, match_info = _find_best_index_match(raw, index_name=index_name, security_col=security_col, label_col=label_col)

    equity = raw.loc[mask, [date_col, price_col]].copy()
    equity = equity.rename(columns={date_col: "date", price_col: "price"})
    equity["date"] = pd.to_datetime(equity["date"], errors="coerce").dt.normalize()
    equity["price"] = pd.to_numeric(equity["price"], errors="coerce")
    equity = equity.dropna(subset=["date", "price"])
    equity = equity.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    equity["log_price"] = np.log(equity["price"])
    equity["log_return"] = equity["log_price"].diff()
    equity["index"] = index_name

    print(f"\nSelected index series: {index_name}")
    print(f"  - Match source: {match_info['match_source']}")
    print(f"  - Match value: {match_info['match_value']}")
    print(f"  - Observations after cleaning: {len(equity):,}")
    print(f"  - Date range: {equity['date'].min().date()} to {equity['date'].max().date()}")

    return equity
