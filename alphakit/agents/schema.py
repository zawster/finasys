"""Schema description generation for LLM system prompts."""

from __future__ import annotations

import polars as pl


def schema(df: pl.DataFrame) -> str:
    """Generate a human-readable schema description of a DataFrame.

    Designed for use in LLM system prompts to help the model understand
    what data it's working with.

    Args:
        df: Any Polars DataFrame.

    Returns:
        A descriptive string suitable for a system prompt.
    """
    parts = []

    # Basic shape
    parts.append(f"DataFrame with {df.height:,} rows and {df.width} columns.")

    # Date range
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        parts.append(f"Time range: {ts.min()} to {ts.max()}.")

    # Symbols
    if "symbol" in df.columns:
        symbols = df["symbol"].unique().sort().to_list()
        if len(symbols) <= 10:
            parts.append(f"Symbols: {', '.join(str(s) for s in symbols)}.")
        else:
            parts.append(f"Symbols: {len(symbols)} unique tickers.")

    # Column descriptions
    col_descs = []
    for col_name in df.columns:
        dtype = df.schema[col_name]
        col_descs.append(f"  {col_name} ({dtype})")

    parts.append("Columns:\n" + "\n".join(col_descs))

    # Data quality notes
    null_cols = []
    for col_name in df.columns:
        null_count = df[col_name].null_count()
        if null_count > 0:
            pct = null_count / df.height * 100
            null_cols.append(f"{col_name}: {null_count} nulls ({pct:.0f}%)")

    if null_cols:
        parts.append("Null values:\n  " + "\n  ".join(null_cols))

    return "\n".join(parts)
