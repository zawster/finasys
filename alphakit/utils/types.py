"""Shared type aliases and protocols for alphakit."""

from __future__ import annotations

from typing import Literal

import polars as pl

# The primary DataFrame type used throughout alphakit
PolarsFrame = pl.DataFrame | pl.LazyFrame

# Backend selection for output format
Backend = Literal["polars", "pandas"]

# Ticker can be a single string or list of strings
Ticker = str | list[str]

# Standard OHLCV column names used throughout alphakit
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
REQUIRED_COLUMNS = ["timestamp", "close"]

# Column name mapping from common formats to alphakit standard
COLUMN_ALIASES: dict[str, str] = {
    "date": "timestamp",
    "datetime": "timestamp",
    "time": "timestamp",
    "Date": "timestamp",
    "Datetime": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "close",
    "adj_close": "close",
    "Volume": "volume",
    "vol": "volume",
    "Vol": "volume",
    "Symbol": "symbol",
    "ticker": "symbol",
    "Ticker": "symbol",
}
