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

## AI Agent Tools

```python
# LLM-ready summary
summary = fs.agents.summarize(df)

# OpenAI function-calling tools
tools = fs.agents.tools(symbols=["AAPL"])

# Context extraction for RAG
context = fs.agents.context(df, "What is the recent momentum?")
```
