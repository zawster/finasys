<p align="center">
  <img src="assets/logo.png" alt="finasys" width="80" style="vertical-align: middle; margin-right: 12px;">
  <span style="font-size: 3rem; font-weight: 700; vertical-align: middle;">finasys</span>
</p>

<p align="center">
  <strong>From raw market data to ML-ready features in five lines of code.</strong>
</p>

finasys is a toolkit for *financial data processing — not manual wrangling — for ML pipelines and AI agents*. It lets you go from **raw market data to production-ready features** in a few lines of code, whether you're building trading models, running portfolio analysis, or powering financial AI agents.

finasys is **Polars-first** — every indicator and feature runs as a native Polars expression, making it 10-100x faster than pandas-based alternatives with **zero C dependencies** (no ta-lib build headaches). It supports **37+ international markets**, crypto, forex, commodities, and macro indicators out of the box.

## Quick Start

```python
import finasys as fs

df = fs.load("AAPL", start="2024-01-01")
df = fs.features.add_all(df)
print(fs.agents.summarize(df))
```

## Install

```bash
pip install finasys
```

## Why finasys?

- **Polars-first** -- 10-100x faster than pandas-ta, zero C dependencies
- **Financial-native** -- every function understands OHLCV, ticks, fundamentals
- **Agent-ready** -- structured outputs designed for LLM consumption
- **Symbol-aware** -- multi-symbol DataFrames work correctly out of the box
