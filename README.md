<p align="center">
  <img src="docs/docs/assets/logo.jpeg" alt="alphakit" width="200">
</p>

<h1 align="center">alphakit</h1>

<p align="center">
  <strong>From raw market data to ML-ready features in five lines of code.</strong>
</p>

<p align="center">
  <a href="https://github.com/zawster/alphakit/actions"><img src="https://github.com/zawster/alphakit/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/zawster/alphakit/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
</p>

A Polars-first Python toolkit for financial data processing, feature engineering, and AI agent integration.

## Quick Start

```python
import alphakit as ak

# Load stock data (auto-cached with DuckDB)
df = ak.load("AAPL", start="2024-01-01")

# Add technical indicators + returns in one call
df = ak.features.add_all(df)

# Generate an LLM-ready summary
print(ak.agents.summarize(df))
```

## Install

```bash
pip install alphakit
```

Optional extras:
```bash
pip install alphakit[langchain]   # LangChain tool integration
pip install alphakit[pandas]      # Pandas interop
pip install alphakit[all]         # Everything
```

## Features

### Data Sources (`alphakit.sources`)
- Single `ak.load()` entry point for Yahoo Finance, CSV, and Parquet files
- Standardized OHLCV column names across all sources
- DuckDB-backed local caching (second call is instant)
- Multi-symbol fetching with automatic alignment

```python
df = ak.load("AAPL", start="2024-01-01")
df = ak.load(["AAPL", "GOOGL", "MSFT"], start="2024-01-01")
df = ak.load("./data/prices.csv")
```

### Feature Engineering (`alphakit.features`)
- 15+ technical indicators: RSI, MACD, Bollinger Bands, ATR, VWAP, OBV, Stochastic, ADX, CCI, Williams %R, MFI, ROC, Momentum
- Returns: simple, log, cumulative, drawdown
- Rolling statistics: mean, std, min, max, skew, z-score
- Lag features with built-in look-ahead bias protection
- Calendar features: day of week, month, quarter
- Cross-sectional: rank, percentile, z-score across symbols

All implemented in **pure Polars expressions** -- no ta-lib C dependency, 10-100x faster than pandas-ta.

```python
# Individual features
df = ak.features.rsi(df, period=14)
df = ak.features.macd(df)
df = ak.features.returns(df, periods=[1, 5, 21])

# Composable pipeline (serializable)
feature_set = ak.FeatureSet([
    ak.features.RSI(period=14),
    ak.features.MACD(),
    ak.features.Returns(periods=[1, 5, 21]),
    ak.features.RollingStats(windows=[5, 21]),
])
df = feature_set.transform(df)
feature_set.save("features.json")
```

### AI Agent Tools (`alphakit.agents`)
- LLM-ready summaries of financial DataFrames
- Tool definitions in OpenAI function-calling format
- Context extraction for RAG-style usage
- Schema descriptions for system prompts
- LangChain integration (optional)

```python
# Summary for LLM context
summary = ak.agents.summarize(df)

# OpenAI function-calling tools
tools = ak.agents.tools(symbols=["AAPL", "GOOGL"])

# LangChain integration
from alphakit.agents.langchain import get_tools
lc_tools = get_tools(symbols=["AAPL"])
```

## Why alphakit?

| | alphakit | pandas-ta | ta-lib |
|---|---|---|---|
| **Engine** | Polars (fast) | pandas (slow) | C library |
| **Install** | `pip install alphakit` | `pip install pandas-ta` | Requires C build tools |
| **AI Agent support** | Built-in | None | None |
| **Caching** | DuckDB auto-cache | None | None |
| **Look-ahead protection** | Built-in | None | None |

## License

MIT
