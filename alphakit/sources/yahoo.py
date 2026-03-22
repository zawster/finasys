"""Yahoo Finance data source via yfinance."""

from __future__ import annotations

from datetime import date, datetime

import polars as pl
import yfinance as yf

from alphakit.sources.schema import cast_ohlcv_types, standardize_columns


def fetch_yahoo(
    symbol: str,
    start: str | date | None = None,
    end: str | date | None = None,
    interval: str = "1d",
) -> pl.DataFrame:
    """Fetch OHLCV data from Yahoo Finance for a single symbol.

    Args:
        symbol: Ticker symbol (e.g., "AAPL").
        start: Start date as string "YYYY-MM-DD" or date object.
        end: End date as string "YYYY-MM-DD" or date object.
        interval: Data interval -- "1d", "1wk", "1mo", etc.

    Returns:
        Polars DataFrame with standardized column names.
    """
    ticker = yf.Ticker(symbol)

    # yfinance returns a pandas DataFrame
    kwargs: dict = {"interval": interval}
    if start is not None:
        kwargs["start"] = str(start)
    if end is not None:
        kwargs["end"] = str(end)

    pdf = ticker.history(**kwargs)

    if pdf.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'. Check the ticker name.")

    # Convert pandas index (DatetimeIndex) to a column
    pdf = pdf.reset_index()

    # Convert to Polars
    df = pl.from_pandas(pdf)

    # Standardize column names (Date -> timestamp, Close -> close, etc.)
    df = standardize_columns(df)

    # Cast to proper types
    df = cast_ohlcv_types(df)

    # Add symbol column
    df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))

    # Keep only standard columns (drop dividends, stock splits, etc.)
    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
            if c in df.columns]
    df = df.select(keep)

    # Ensure timestamp is Date type for daily data
    if interval == "1d" and "timestamp" in df.columns:
        ts_type = df.schema["timestamp"]
        if ts_type == pl.Datetime or str(ts_type).startswith("Datetime"):
            df = df.with_columns(pl.col("timestamp").cast(pl.Date))

    return df


def fetch_yahoo_multi(
    symbols: list[str],
    start: str | date | None = None,
    end: str | date | None = None,
    interval: str = "1d",
) -> pl.DataFrame:
    """Fetch OHLCV data for multiple symbols and stack into one DataFrame.

    Each row has a 'symbol' column identifying the ticker.
    """
    frames = []
    errors = []

    for sym in symbols:
        try:
            df = fetch_yahoo(sym, start=start, end=end, interval=interval)
            frames.append(df)
        except ValueError as e:
            errors.append(str(e))

    if not frames:
        raise ValueError(
            f"No data returned for any symbols. Errors: {'; '.join(errors)}"
        )

    result = pl.concat(frames, how="vertical_relaxed")
    return result.sort(["timestamp", "symbol"])
