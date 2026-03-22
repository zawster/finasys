"""alphakit.sources -- Unified financial data ingestion.

The main entry point is `load()`, which dispatches to the appropriate
data source based on the input:

    import alphakit as ak

    # Single ticker from Yahoo Finance
    df = ak.load("AAPL", start="2024-01-01")

    # Multiple tickers
    df = ak.load(["AAPL", "GOOGL", "MSFT"], start="2024-01-01")

    # Local file (CSV or Parquet)
    df = ak.load("./data/prices.csv")
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from alphakit.sources.cache import cache_clear, cache_get, cache_put
from alphakit.sources.local import load_local
from alphakit.sources.yahoo import fetch_yahoo, fetch_yahoo_multi
from alphakit.utils.types import Backend, Ticker


def load(
    source: Ticker | str | Path,
    start: str | date | None = None,
    end: str | date | None = None,
    interval: str = "1d",
    backend: Backend = "polars",
    use_cache: bool = True,
) -> pl.DataFrame:
    """Load financial data from any supported source.

    This is the single entry point for all data loading in alphakit.
    It auto-detects whether the source is a ticker symbol, list of tickers,
    or a local file path, and dispatches accordingly.

    Args:
        source: A ticker string ("AAPL"), list of tickers (["AAPL", "GOOGL"]),
                or a file path ("./data/prices.csv").
        start: Start date for Yahoo Finance fetches.
        end: End date for Yahoo Finance fetches.
        interval: Data interval for Yahoo Finance ("1d", "1wk", "1mo").
        backend: Output format -- "polars" (default) or "pandas".
        use_cache: Whether to use DuckDB caching for Yahoo fetches.

    Returns:
        A Polars DataFrame (or pandas DataFrame if backend="pandas")
        with standardized column names: timestamp, open, high, low, close,
        volume, symbol.
    """
    # Determine what kind of source this is
    if isinstance(source, list):
        df = _load_multi_ticker(source, start, end, interval, use_cache)
    elif isinstance(source, Path) or (isinstance(source, str) and _is_file_path(source)):
        df = load_local(source)
    else:
        df = _load_single_ticker(str(source), start, end, interval, use_cache)

    # Convert to pandas if requested
    if backend == "pandas":
        return df.to_pandas()  # type: ignore[return-value]

    return df


def _is_file_path(s: str) -> bool:
    """Heuristic to detect if a string is a file path vs. a ticker symbol."""
    # Contains path separators or file extensions
    if "/" in s or "\\" in s:
        return True
    if "." in s:
        suffix = s.rsplit(".", 1)[-1].lower()
        return suffix in ("csv", "parquet", "pq", "json")
    return False


def _load_single_ticker(
    symbol: str,
    start: str | date | None,
    end: str | date | None,
    interval: str,
    use_cache: bool,
) -> pl.DataFrame:
    """Load data for a single ticker, with caching."""
    if use_cache:
        cached = cache_get(symbol, start, end)
        if cached is not None and not cached.is_empty():
            return cached

    df = fetch_yahoo(symbol, start=start, end=end, interval=interval)

    if use_cache:
        cache_put(df, symbol)

    return df


def _load_multi_ticker(
    symbols: list[str],
    start: str | date | None,
    end: str | date | None,
    interval: str,
    use_cache: bool,
) -> pl.DataFrame:
    """Load data for multiple tickers."""
    if use_cache:
        frames = []
        to_fetch = []
        for sym in symbols:
            cached = cache_get(sym, start, end)
            if cached is not None and not cached.is_empty():
                frames.append(cached)
            else:
                to_fetch.append(sym)

        if to_fetch:
            fresh = fetch_yahoo_multi(to_fetch, start=start, end=end, interval=interval)
            for sym in to_fetch:
                sym_data = fresh.filter(pl.col("symbol") == sym.upper())
                if not sym_data.is_empty():
                    cache_put(sym_data, sym)
            frames.append(fresh)

        if frames:
            return pl.concat(frames, how="vertical_relaxed").sort(["timestamp", "symbol"])
        return pl.DataFrame()

    return fetch_yahoo_multi(symbols, start=start, end=end, interval=interval)


__all__ = ["load", "cache_clear"]
