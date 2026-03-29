# Quickstart

## Installation

```bash
pip install finasys
```

## Load Data

```python
import finasys as fs

# Single symbol
df = fs.load("AAPL", start="2024-01-01")

# Multiple symbols
df = fs.load(["AAPL", "GOOGL", "MSFT"], start="2024-01-01")

# Local files
df = fs.load("./data/prices.csv")
```

## Add Features

```python
# Individual indicators
df = fs.features.rsi(df, period=14)
df = fs.features.macd(df)
df = fs.features.bollinger(df)

# Or add everything at once
df = fs.features.add_all(df)
```

## Composable Pipeline

```python
feature_set = fs.FeatureSet([
    fs.features.RSI(period=14),
    fs.features.MACD(),
    fs.features.Returns(periods=[1, 5, 21]),
])

df = feature_set.transform(df)
feature_set.save("my_pipeline.json")  # reproducible
```

## ML Targets

```python
# Forward returns for regression
df = fs.features.forward_returns(df, periods=[1, 5])

# Classification labels
df = fs.features.classify_returns(df, period=5)

# Triple-barrier labeling (Lopez de Prado method)
df = fs.features.triple_barrier_labels(df, profit_take=0.02, stop_loss=0.02)
```

## Distribution Features

```python
df = fs.features.rolling_kurtosis(df, window=30)
df = fs.features.zscore_returns(df, window=30)
df = fs.features.tail_ratio(df, window=30)
```

## Risk Metrics

```python
# Scalar metrics
sharpe = fs.stats.sharpe_ratio(df)
var = fs.stats.value_at_risk(df, confidence=0.95)

# Rolling metrics (as ML features)
df = fs.stats.sharpe_ratio(df, window=63)
```

## Smart Profiler

```python
# One-call data quality check
print(fs.profiler.profile_summary(df))

# Full structured report
report = fs.profiler.profile(df)
```

## AI Agent Tools

```python
# LLM-ready summary
summary = fs.agents.summarize(df)

# OpenAI function-calling tools
tools = fs.agents.tools(symbols=["AAPL"])

# Context extraction for RAG
context = fs.agents.context(df, "What is the recent momentum?")
```
