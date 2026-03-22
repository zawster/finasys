"""Return calculations for financial time series."""

from __future__ import annotations

import numpy as np
import polars as pl

from alphakit.features.utils import has_multi_symbols, symbol_aware


def returns(
    df: pl.DataFrame,
    periods: list[int] | int = 1,
    column: str = "close",
) -> pl.DataFrame:
    """Simple percentage returns over given period(s).

    Symbol-aware: in multi-symbol DataFrames, returns are computed
    within each symbol (no cross-contamination).

    Args:
        df: DataFrame with price data.
        periods: Single period or list of periods (e.g., [1, 5, 21]
                 for daily, weekly, monthly returns).
        column: Price column to compute returns from.

    Returns:
        DataFrame with return columns appended (e.g., returns_1d, returns_5d).
    """
    if isinstance(periods, int):
        periods = [periods]

    col = pl.col(column)
    exprs = []
    for p in periods:
        expr = (col / col.shift(p) - 1).alias(f"returns_{p}d")
        exprs.append(symbol_aware(expr, df))

    return df.with_columns(exprs)


def log_returns(
    df: pl.DataFrame,
    periods: list[int] | int = 1,
    column: str = "close",
) -> pl.DataFrame:
    """Log returns over given period(s).

    Log returns are additive over time, making them useful for
    statistical analysis and portfolio calculations.
    Symbol-aware for multi-symbol DataFrames.
    """
    if isinstance(periods, int):
        periods = [periods]

    col = pl.col(column)
    exprs = []
    for p in periods:
        expr = (col / col.shift(p)).log().alias(f"log_returns_{p}d")
        exprs.append(symbol_aware(expr, df))

    return df.with_columns(exprs)


def cumulative_returns(
    df: pl.DataFrame, column: str = "close"
) -> pl.DataFrame:
    """Cumulative returns from the first data point.

    Calculated as (current_price / first_price) - 1.
    Symbol-aware for multi-symbol DataFrames.
    """
    col = pl.col(column)
    expr = (col / col.first() - 1).alias("cumulative_returns")
    return df.with_columns(symbol_aware(expr, df))


def drawdown(df: pl.DataFrame, column: str = "close") -> pl.DataFrame:
    """Drawdown from running maximum.

    Measures the decline from the peak price. Useful for risk assessment.
    Symbol-aware for multi-symbol DataFrames.
    Appends two columns:
    - drawdown: percentage decline from peak (always <= 0)
    - max_drawdown: running maximum drawdown
    """
    col = pl.col(column)
    running_max = col.cum_max()

    dd = (col - running_max) / running_max

    dd_expr = symbol_aware(dd.alias("drawdown"), df)
    max_dd_expr = symbol_aware(dd.cum_min().alias("max_drawdown"), df)

    return df.with_columns(dd_expr, max_dd_expr)
