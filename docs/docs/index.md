# alphakit

**From raw market data to ML-ready features in five lines of code.**

alphakit is a Polars-first Python toolkit for financial data processing, feature engineering, and AI agent integration.

## Quick Start

```python
import alphakit as ak

df = ak.load("AAPL", start="2024-01-01")
df = ak.features.add_all(df)
print(ak.agents.summarize(df))
```

## Install

```bash
pip install alphakit
```

## Why alphakit?

- **Polars-first** -- 10-100x faster than pandas-ta, zero C dependencies
- **Financial-native** -- every function understands OHLCV, ticks, fundamentals
- **Agent-ready** -- structured outputs designed for LLM consumption
- **Symbol-aware** -- multi-symbol DataFrames work correctly out of the box
