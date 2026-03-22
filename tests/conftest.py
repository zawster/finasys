"""Shared test fixtures for alphakit tests.

All fixtures use synthetic data -- no network calls required.
"""

from datetime import date, timedelta

import polars as pl
import pytest


@pytest.fixture
def ohlcv_df() -> pl.DataFrame:
    """A synthetic OHLCV DataFrame with 100 trading days.

    Simulates a stock that starts at $100 and has realistic
    daily price movements.
    """
    n = 100
    start_date = date(2024, 1, 2)

    # Generate dates (skip weekends roughly)
    dates = []
    d = start_date
    while len(dates) < n:
        if d.weekday() < 5:  # Mon-Fri
            dates.append(d)
        d += timedelta(days=1)

    # Generate realistic OHLCV data
    import numpy as np

    rng = np.random.RandomState(42)
    close_prices = [100.0]
    for _ in range(n - 1):
        # Random walk with slight upward drift
        change = rng.normal(0.0005, 0.015)
        close_prices.append(close_prices[-1] * (1 + change))

    close = np.array(close_prices)
    # High is close + random positive amount
    high = close + rng.uniform(0.5, 2.0, n)
    # Low is close - random positive amount
    low = close - rng.uniform(0.5, 2.0, n)
    # Open is between low and high
    open_ = low + rng.uniform(0.3, 0.7, n) * (high - low)
    # Volume between 1M and 10M
    volume = rng.uniform(1_000_000, 10_000_000, n)

    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": open_.tolist(),
            "high": high.tolist(),
            "low": low.tolist(),
            "close": close.tolist(),
            "volume": volume.tolist(),
            "symbol": ["AAPL"] * n,
        }
    ).cast(
        {
            "timestamp": pl.Date,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        }
    )


@pytest.fixture
def multi_symbol_df(ohlcv_df: pl.DataFrame) -> pl.DataFrame:
    """A multi-symbol DataFrame with AAPL and GOOGL."""
    import numpy as np

    rng = np.random.RandomState(123)

    googl = ohlcv_df.clone()
    # Adjust prices for GOOGL (different price level)
    googl = googl.with_columns(
        [
            (pl.col("open") * 1.5 + rng.normal(0, 1, ohlcv_df.height)).alias("open"),
            (pl.col("high") * 1.5 + rng.normal(0, 1, ohlcv_df.height)).alias("high"),
            (pl.col("low") * 1.5 + rng.normal(0, 1, ohlcv_df.height)).alias("low"),
            (pl.col("close") * 1.5 + rng.normal(0, 1, ohlcv_df.height)).alias("close"),
            pl.lit("GOOGL").alias("symbol"),
        ]
    )

    return pl.concat([ohlcv_df, googl]).sort(["timestamp", "symbol"])


@pytest.fixture
def simple_close_df() -> pl.DataFrame:
    """A minimal DataFrame with just timestamp and close for simple tests."""
    return pl.DataFrame(
        {
            "timestamp": [date(2024, 1, i) for i in range(1, 11)],
            "close": [100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 108.0, 107.0, 110.0],
        }
    )
