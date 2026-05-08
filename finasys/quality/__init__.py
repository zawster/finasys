"""Financial data quality checks."""

from __future__ import annotations

from datetime import date as dt_date
from typing import Any

import numpy as np
import polars as pl

from finasys.features.utils import has_multi_symbols, symbol_aware

__all__ = [
    "detect_gaps",
    "flag_outliers",
    "detect_splits",
    "completeness_report",
]


def _business_day_gaps(df: pl.DataFrame, timestamp_col: str = "timestamp") -> list[str]:
    if timestamp_col not in df.columns or df.height < 2:
        return []

    ts = df[timestamp_col].sort()
    start = ts.min()
    end = ts.max()
    if start is None or end is None:
        return []

    all_days = np.arange(
        np.datetime64(start),
        np.datetime64(end) + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    business_days = all_days[np.is_busday(all_days)]
    existing = set(ts.to_list())
    missing = []
    for bd in business_days:
        day = bd.astype("datetime64[D]").astype(dt_date)
        if day not in existing:
            missing.append(str(day))
    return missing


def detect_gaps(
    df: pl.DataFrame,
    freq: str = "1bd",
    timestamp_col: str = "timestamp",
) -> list[str] | dict[str, list[str]]:
    """Find missing dates in a financial time series.

    Currently supports business-day frequency via "1bd". For multi-symbol
    DataFrames, returns a mapping of symbol to missing date strings.
    """
    if freq != "1bd":
        raise ValueError("Only business-day frequency '1bd' is currently supported")

    if has_multi_symbols(df):
        return {
            sym: _business_day_gaps(df.filter(pl.col("symbol") == sym), timestamp_col)
            for sym in df["symbol"].unique().sort().to_list()
        }
    return _business_day_gaps(df, timestamp_col)


def flag_outliers(
    df: pl.DataFrame,
    method: str = "zscore",
    threshold: float = 4.0,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Append boolean outlier flags for numeric columns.

    Args:
        method: "zscore" or "iqr".
        threshold: Z-score threshold, or IQR multiplier.
        columns: Columns to check. Defaults to available OHLC columns.
    """
    if method not in ("zscore", "iqr"):
        raise ValueError("method must be 'zscore' or 'iqr'")

    if columns is None:
        columns = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if not columns:
        return df

    exprs = []
    for col_name in columns:
        col = pl.col(col_name)
        if method == "zscore":
            mean = col.mean()
            std = col.std()
            expr = ((col - mean).abs() > threshold * std).fill_null(False).alias(f"{col_name}_outlier")
        else:
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            iqr = q3 - q1
            expr = (
                ((col < q1 - threshold * iqr) | (col > q3 + threshold * iqr))
                .fill_null(False)
                .alias(f"{col_name}_outlier")
            )
        exprs.append(symbol_aware(expr, df))

    return df.with_columns(exprs)


def detect_splits(
    df: pl.DataFrame,
    threshold: float = 0.3,
    column: str = "close",
) -> pl.DataFrame:
    """Append a suspected_split boolean column based on large price jumps."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    pct_change = (pl.col(column) / pl.col(column).shift(1) - 1).abs()
    expr = (pct_change > threshold).fill_null(False).alias("suspected_split")
    return df.with_columns(symbol_aware(expr, df))


def completeness_report(df: pl.DataFrame) -> dict[str, Any]:
    """Return a JSON-serializable completeness and quality summary."""
    gaps = detect_gaps(df) if "timestamp" in df.columns else []
    gap_count = sum(len(v) for v in gaps.values()) if isinstance(gaps, dict) else len(gaps)
    duplicate_rows = int(df.is_duplicated().sum())
    zero_volume_days = df.filter(pl.col("volume") == 0).height if "volume" in df.columns else 0
    null_counts = {col: int(df[col].null_count()) for col in df.columns}
    null_pct = {col: (count / df.height * 100 if df.height else 0.0) for col, count in null_counts.items()}

    split_count = 0
    if "close" in df.columns:
        split_count = detect_splits(df)["suspected_split"].sum()

    outlier_counts: dict[str, int] = {}
    flagged = flag_outliers(df) if any(c in df.columns for c in ["open", "high", "low", "close"]) else df
    for col in flagged.columns:
        if col.endswith("_outlier"):
            outlier_counts[col] = int(flagged[col].sum())

    return {
        "rows": df.height,
        "columns": df.width,
        "date_gaps": gaps,
        "date_gap_count": int(gap_count),
        "duplicate_rows": duplicate_rows,
        "zero_volume_days": int(zero_volume_days),
        "null_counts": null_counts,
        "null_pct": null_pct,
        "outlier_counts": outlier_counts,
        "suspected_split_count": int(split_count),
    }
