"""Standardized column schemas for financial data."""

from __future__ import annotations

import polars as pl

from alphakit.utils.types import COLUMN_ALIASES, OHLCV_COLUMNS


def standardize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Rename columns to alphakit's standard lowercase snake_case names.

    Maps common column name variations (Date, Close, Adj Close, etc.)
    to the standard: timestamp, open, high, low, close, volume, symbol.
    """
    rename_map = {}
    for col in df.columns:
        if col in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[col]
        elif col != col.lower():
            # Also lowercase any remaining uppercase column names
            lower = col.lower().replace(" ", "_")
            if lower != col:
                rename_map[col] = lower

    if rename_map:
        df = df.rename(rename_map)

    return df


def validate_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    """Validate that a DataFrame has the minimum required columns for OHLCV data.

    Raises ValueError if 'timestamp' and 'close' columns are missing.
    """
    missing = [c for c in ["timestamp", "close"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {df.columns}. "
            f"Use alphakit's standard names: {OHLCV_COLUMNS}"
        )
    return df


def detect_ohlcv_schema(df: pl.DataFrame) -> bool:
    """Check if a DataFrame looks like OHLCV data (has at least timestamp + close)."""
    std = standardize_columns(df)
    return "timestamp" in std.columns and "close" in std.columns


def cast_ohlcv_types(df: pl.DataFrame) -> pl.DataFrame:
    """Cast OHLCV columns to appropriate types.

    - timestamp -> Date or Datetime (auto-detected)
    - open/high/low/close -> Float64
    - volume -> Int64 or Float64
    """
    casts = {}

    if "timestamp" in df.columns:
        col_type = df.schema["timestamp"]
        if col_type == pl.Utf8 or col_type == pl.String:
            # Try to parse as date/datetime
            try:
                df = df.with_columns(pl.col("timestamp").str.to_date().alias("timestamp"))
            except Exception:
                try:
                    df = df.with_columns(
                        pl.col("timestamp").str.to_datetime().alias("timestamp")
                    )
                except Exception:
                    pass  # Leave as string if parsing fails

    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    if price_cols:
        casts.update({c: pl.Float64 for c in price_cols})

    if "volume" in df.columns:
        casts["volume"] = pl.Float64

    if casts:
        df = df.cast(casts, strict=False)

    return df
