"""Shared constants for finasys."""

# Annualization factor (trading days per year)
TRADING_DAYS = 252

# Standard OHLCV column names used throughout finasys
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
REQUIRED_COLUMNS = ["timestamp", "close"]

# Column name mapping from common formats to finasys standard
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
