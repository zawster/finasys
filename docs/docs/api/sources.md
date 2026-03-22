# Sources API

## `ak.load()`

The single entry point for all data loading in alphakit.

```python
import alphakit as ak

# Single ticker from Yahoo Finance
df = ak.load("AAPL", start="2024-01-01")

# Multiple tickers (stacked with 'symbol' column)
df = ak.load(["AAPL", "GOOGL", "MSFT"], start="2024-01-01")

# Local CSV or Parquet file
df = ak.load("./data/prices.csv")

# With date range
df = ak.load("AAPL", start="2024-01-01", end="2024-12-31")

# Get pandas DataFrame instead of Polars
df = ak.load("AAPL", start="2024-01-01", backend="pandas")

# Disable caching
df = ak.load("AAPL", start="2024-01-01", use_cache=False)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str`, `list[str]`, or `Path` | required | Ticker symbol, list of tickers, or file path |
| `start` | `str` or `date` | `None` | Start date (e.g., `"2024-01-01"`) |
| `end` | `str` or `date` | `None` | End date |
| `interval` | `str` | `"1d"` | Data interval: `"1d"`, `"1wk"`, `"1mo"` |
| `backend` | `str` | `"polars"` | Output format: `"polars"` or `"pandas"` |
| `use_cache` | `bool` | `True` | Use DuckDB local cache |

**Returns:** Polars DataFrame with standardized columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`, `symbol`

**Supported sources:**

- Yahoo Finance tickers: `"AAPL"`, `"^GSPC"`, `"BTC-USD"`, `"EURUSD=X"`, `"GC=F"`, etc.
- Local files: `.csv`, `.parquet`, `.json`

---

## `ak.cache_clear()`

Clear the DuckDB local cache.

```python
ak.cache_clear()          # Clear all cached data
ak.cache_clear("AAPL")    # Clear only AAPL data
```

---

## Standardized Column Names

alphakit normalizes all column names to a standard format:

| Input Variations | alphakit Standard |
|------------------|-------------------|
| `Date`, `Datetime`, `time` | `timestamp` |
| `Open` | `open` |
| `High` | `high` |
| `Low` | `low` |
| `Close`, `Adj Close` | `close` |
| `Volume`, `Vol` | `volume` |
| `Ticker`, `Symbol` | `symbol` |
