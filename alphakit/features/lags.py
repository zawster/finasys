"""Lag and lead feature generation with look-ahead bias protection."""

from __future__ import annotations

import polars as pl

from alphakit.features.utils import symbol_aware


def lags(
    df: pl.DataFrame,
    columns: list[str] | str = "close",
    lags: list[int] | int = 1,
) -> pl.DataFrame:
    """Create lagged features (shifted forward in time = looking backward).

    Lag features use PAST data only, so they are safe from look-ahead bias.
    A lag of 1 means "yesterday's value".

    Symbol-aware: in multi-symbol DataFrames, lags don't cross symbol boundaries.

    Args:
        df: Input DataFrame.
        columns: Column(s) to create lags for.
        lags: Lag period(s). Positive values look backward (safe).

    Returns:
        DataFrame with lag columns appended (e.g., close_lag_1, close_lag_5).
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(lags, int):
        lags = [lags]

    # Safety check: all lags must be positive to avoid look-ahead bias
    if any(lag <= 0 for lag in lags):
        raise ValueError(
            "All lag values must be positive (look backward in time). "
            "Negative or zero lags would cause look-ahead bias."
        )

    exprs = []
    for col in columns:
        for lag in lags:
            expr = pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")
            exprs.append(symbol_aware(expr, df))

    return df.with_columns(exprs)


def validate_no_lookahead(
    df_full: pl.DataFrame,
    df_partial: pl.DataFrame,
    feature_columns: list[str],
) -> bool:
    """Validate that features don't use future data.

    Compares features computed on the full dataset vs. a truncated dataset.
    If features at overlapping timestamps differ, there's look-ahead bias.

    Args:
        df_full: Features computed on the complete dataset.
        df_partial: Features computed on a truncated (earlier) subset.
        feature_columns: Columns to check for look-ahead bias.

    Returns:
        True if no look-ahead bias detected, False otherwise.
    """
    # Get overlapping timestamps
    partial_ts = df_partial.select("timestamp")
    overlap = df_full.join(partial_ts, on="timestamp", how="semi")

    # Compare feature values at overlapping timestamps
    for col in feature_columns:
        if col not in overlap.columns or col not in df_partial.columns:
            continue

        full_vals = overlap.select("timestamp", col).sort("timestamp")
        partial_vals = df_partial.select("timestamp", col).sort("timestamp")

        # Join and compare
        comparison = full_vals.join(partial_vals, on="timestamp", suffix="_partial")

        col_full = pl.col(col)
        col_partial = pl.col(f"{col}_partial")

        # Check if values match (allowing for NaN == NaN)
        mismatches = comparison.filter(
            col_full.is_not_null()
            & col_partial.is_not_null()
            & ((col_full - col_partial).abs() > 1e-10)
        )

        if mismatches.height > 0:
            return False

    return True
