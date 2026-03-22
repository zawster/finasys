"""DuckDB-backed local cache for financial data."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from alphakit.utils.config import config


def _get_db_path() -> Path:
    """Get the path to the DuckDB cache file."""
    cache_dir = config.ensure_cache_dir()
    return cache_dir / "alphakit_cache.duckdb"


def _get_connection() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection and ensure the cache table exists."""
    db_path = _get_db_path()
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_cache (
            symbol VARCHAR,
            timestamp DATE,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            PRIMARY KEY (symbol, timestamp)
        )
    """)
    return conn


def cache_get(
    symbol: str,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pl.DataFrame | None:
    """Retrieve cached data for a symbol within a date range.

    Returns None if no cached data exists for the given parameters.
    """
    if not config.cache_enabled:
        return None

    try:
        conn = _get_connection()
    except Exception:
        return None

    query = "SELECT * FROM ohlcv_cache WHERE symbol = ?"
    params: list = [symbol.upper()]

    if start is not None:
        query += " AND timestamp >= ?"
        params.append(str(start))
    if end is not None:
        query += " AND timestamp <= ?"
        params.append(str(end))

    query += " ORDER BY timestamp"

    try:
        result = conn.execute(query, params).pl()
        conn.close()

        if result.is_empty():
            return None
        return result
    except Exception:
        conn.close()
        return None


def cache_put(df: pl.DataFrame, symbol: str) -> None:
    """Store OHLCV data in the cache. Upserts on (symbol, timestamp)."""
    if not config.cache_enabled:
        return

    if df.is_empty():
        return

    try:
        conn = _get_connection()
    except Exception:
        return

    # Always set the symbol column to the cache key (overrides any existing symbol)
    store_df = df.clone()
    store_df = store_df.with_columns(pl.lit(symbol.upper()).alias("symbol"))

    # Select only the columns we cache
    cache_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
    available = [c for c in cache_cols if c in store_df.columns]
    store_df = store_df.select(available)

    try:
        # Use INSERT OR REPLACE for upsert behavior
        conn.execute("DELETE FROM ohlcv_cache WHERE symbol = ?", [symbol.upper()])
        conn.execute("INSERT INTO ohlcv_cache SELECT * FROM store_df")
        conn.close()
    except Exception:
        conn.close()


def cache_clear(symbol: str | None = None) -> None:
    """Clear cached data. If symbol is given, clear only that symbol."""
    try:
        conn = _get_connection()
        if symbol is None:
            conn.execute("DELETE FROM ohlcv_cache")
        else:
            conn.execute("DELETE FROM ohlcv_cache WHERE symbol = ?", [symbol.upper()])
        conn.close()
    except Exception:
        pass
