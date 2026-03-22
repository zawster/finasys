"""Shared utilities for feature functions."""

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
