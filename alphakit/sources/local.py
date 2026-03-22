"""Local file loaders (CSV, Parquet) with financial schema auto-detection."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from alphakit.sources.schema import (
    cast_ohlcv_types,
    detect_ohlcv_schema,
    standardize_columns,
)


def load_local(path: str | Path) -> pl.DataFrame:
    """Load a local CSV or Parquet file and standardize to OHLCV schema.

    Automatically detects file format by extension, renames columns
    to alphakit standard names, and casts types.

    Args:
        path: Path to a CSV or Parquet file.

    Returns:
        Polars DataFrame with standardized columns.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is unsupported.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pl.read_csv(path, try_parse_dates=True)
    elif suffix in (".parquet", ".pq"):
        df = pl.read_parquet(path)
    elif suffix == ".json":
        df = pl.read_json(path)
    else:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported formats: .csv, .parquet, .pq, .json"
        )

    # Standardize column names
    df = standardize_columns(df)

    # Auto-detect if this is OHLCV data and cast types
    if detect_ohlcv_schema(df):
        df = cast_ohlcv_types(df)

    return df
