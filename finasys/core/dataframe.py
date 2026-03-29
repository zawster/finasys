"""Shared DataFrame utilities for finasys."""

from __future__ import annotations

import polars as pl


def has_multi_symbols(df: pl.DataFrame) -> bool:
    """Check if a DataFrame contains multiple symbols."""
    if "symbol" not in df.columns:
        return False
    return df["symbol"].n_unique() > 1


def symbol_aware(expr: pl.Expr, df: pl.DataFrame) -> pl.Expr:
    """Wrap an expression with .over('symbol') if the DataFrame has multiple symbols.

    This ensures that rolling/shift operations don't cross symbol boundaries.
    """
    if has_multi_symbols(df):
        return expr.over("symbol")
    return expr


def apply_per_symbol(
    df: pl.DataFrame,
    func,
    *args,
    **kwargs,
) -> pl.DataFrame:
    """Apply a function per-symbol if multi-symbol, otherwise directly.

    The function must accept a DataFrame as its first argument and return a DataFrame.
    """
    if has_multi_symbols(df):
        frames = []
        for sym in df["symbol"].unique().sort().to_list():
            sym_df = df.filter(pl.col("symbol") == sym)
            sym_df = func(sym_df, *args, **kwargs)
            frames.append(sym_df)
        return pl.concat(frames).sort(["timestamp", "symbol"])
    else:
        return func(df, *args, **kwargs)
