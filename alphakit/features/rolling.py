"""Rolling window statistics for financial time series."""

from __future__ import annotations

import polars as pl

from alphakit.features.utils import symbol_aware


def rolling_stats(
    df: pl.DataFrame,
    windows: list[int] | int = 21,
    column: str = "close",
    stats: list[str] | None = None,
) -> pl.DataFrame:
    """Compute rolling statistics for a given column.

    Symbol-aware: in multi-symbol DataFrames, stats are computed
    within each symbol.

    Args:
        df: Input DataFrame.
        windows: Window size(s) for rolling calculations.
        column: Column to compute stats on.
        stats: Which statistics to compute. Options:
               "mean", "std", "min", "max", "skew", "zscore".
               Defaults to ["mean", "std"].

    Returns:
        DataFrame with rolling stat columns appended.
    """
    if isinstance(windows, int):
        windows = [windows]
    if stats is None:
        stats = ["mean", "std"]

    col = pl.col(column)
    exprs = []

    for w in windows:
        for stat in stats:
            if stat == "mean":
                expr = col.rolling_mean(window_size=w).alias(f"rolling_mean_{w}")
            elif stat == "std":
                expr = col.rolling_std(window_size=w).alias(f"rolling_std_{w}")
            elif stat == "min":
                expr = col.rolling_min(window_size=w).alias(f"rolling_min_{w}")
            elif stat == "max":
                expr = col.rolling_max(window_size=w).alias(f"rolling_max_{w}")
            elif stat == "skew":
                expr = col.rolling_skew(window_size=w).alias(f"rolling_skew_{w}")
            elif stat == "zscore":
                mean = col.rolling_mean(window_size=w)
                std = col.rolling_std(window_size=w)
                expr = ((col - mean) / std).alias(f"rolling_zscore_{w}")
            else:
                raise ValueError(
                    f"Unknown stat: '{stat}'. "
                    f"Options: mean, std, min, max, skew, zscore"
                )
            exprs.append(symbol_aware(expr, df))

    return df.with_columns(exprs)
