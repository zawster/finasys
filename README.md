<p align="center">
  <img src="docs/docs/assets/logo.png" alt="finasys" width="200">
</p>
<h1 align="center">finasys</h1>

<p align="center">
  <strong>From raw market data to ML-ready features in five lines of code.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/finasys/"><img src="https://img.shields.io/pypi/v/finasys.svg" alt="PyPI"></a>
  <a href="https://github.com/zawster/finasys/actions"><img src="https://github.com/zawster/finasys/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/zawster/finasys"><img src="https://codecov.io/gh/zawster/finasys/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://github.com/zawster/finasys/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
</p>

**Documentation:** [finasys Docs](https://zawster.github.io/finasys)

---

finasys is a toolkit for *financial data processing — not manual wrangling — for ML pipelines and AI agents*. It lets you go from **raw market data to production-ready features** in a few lines of code, whether you're building trading models, running portfolio analysis, or powering financial AI agents.

finasys is **Polars-first** — every indicator and feature runs as a native Polars expression, making it 10-100x faster than pandas-based alternatives with **zero C dependencies** (no ta-lib build headaches). It supports **37+ international markets**, crypto, forex, commodities, and macro indicators out of the box. Learn more via our [official documentation](https://zawster.github.io/finasys) or start contributing via this GitHub repo.

## Quick Start

```python
import finasys as fs

# Load stock data (auto-cached with DuckDB)
df = fs.load("AAPL", start="2024-01-01")

# Add technical indicators + returns in one call
df = fs.features.add_all(df)

# Generate an LLM-ready summary
print(fs.agents.summarize(df))
```

## Install

```bash
pip install finasys
```

Optional extras:
```bash
pip install finasys[langchain]   # LangChain tool integration
pip install finasys[pandas]      # Pandas interop
pip install finasys[all]         # Everything
```

## Features

### Data Sources (`finasys.sources`)
- Single `fs.load()` entry point for Yahoo Finance, CSV, and Parquet files
- Standardized OHLCV column names across all sources
- DuckDB-backed local caching (second call is instant)
- Multi-symbol fetching with automatic alignment

```python
df = fs.load("AAPL", start="2024-01-01")
df = fs.load(["AAPL", "GOOGL", "MSFT"], start="2024-01-01")
df = fs.load("./data/prices.csv")
```

### Feature Engineering (`finasys.features`)
- 15+ technical indicators: RSI, MACD, Bollinger Bands, ATR, VWAP, OBV, Stochastic, ADX, CCI, Williams %R, MFI, ROC, Momentum
- Returns: simple, log, cumulative, drawdown
- Rolling statistics: mean, std, min, max, skew, z-score
- Lag features with built-in look-ahead bias protection
- Calendar features: day of week, month, quarter
- Cross-sectional: rank, percentile, z-score across symbols

All implemented in **pure Polars expressions** -- no ta-lib C dependency, 10-100x faster than pandas-ta.

```python
# Individual features
df = fs.features.rsi(df, period=14)
df = fs.features.macd(df)
df = fs.features.returns(df, periods=[1, 5, 21])

# Composable pipeline (serializable)
feature_set = fs.FeatureSet([
    fs.features.RSI(period=14),
    fs.features.MACD(),
    fs.features.Returns(periods=[1, 5, 21]),
    fs.features.RollingStats(windows=[5, 21]),
])
df = feature_set.transform(df)
feature_set.save("features.json")
```

### AI Agent Tools (`finasys.agents`)
- LLM-ready summaries of financial DataFrames
- Tool definitions in OpenAI function-calling format
- Context extraction for RAG-style usage
- Schema descriptions for system prompts
- LangChain integration (optional)

```python
# Summary for LLM context
summary = fs.agents.summarize(df)

# OpenAI function-calling tools
tools = fs.agents.tools(symbols=["AAPL", "GOOGL"])

# LangChain integration
from finasys.agents.langchain import get_tools
lc_tools = get_tools(symbols=["AAPL"])
```

## Why finasys?

| | finasys | pandas-ta | ta-lib |
|---|---|---|---|
| **Engine** | Polars (fast) | pandas (slow) | C library |
| **Install** | `pip install finasys` | `pip install pandas-ta` | Requires C build tools |
| **AI Agent support** | Built-in | None | None |
| **Caching** | DuckDB auto-cache | None | None |
| **Look-ahead protection** | Built-in | None | None |

## License

MIT
