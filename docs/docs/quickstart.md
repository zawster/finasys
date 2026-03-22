# Quickstart

## Installation

```bash
pip install alphakit
```

## Load Data

```python
import alphakit as ak

# Single symbol
df = ak.load("AAPL", start="2024-01-01")

# Multiple symbols
df = ak.load(["AAPL", "GOOGL", "MSFT"], start="2024-01-01")

# Local files
df = ak.load("./data/prices.csv")
```

## Add Features

```python
# Individual indicators
df = ak.features.rsi(df, period=14)
df = ak.features.macd(df)
df = ak.features.bollinger(df)

# Or add everything at once
df = ak.features.add_all(df)
```

## Composable Pipeline

```python
feature_set = ak.FeatureSet([
    ak.features.RSI(period=14),
    ak.features.MACD(),
    ak.features.Returns(periods=[1, 5, 21]),
])

df = feature_set.transform(df)
feature_set.save("my_pipeline.json")  # reproducible
```

## AI Agent Tools

```python
# LLM-ready summary
summary = ak.agents.summarize(df)

# OpenAI function-calling tools
tools = ak.agents.tools(symbols=["AAPL"])

# Context extraction for RAG
context = ak.agents.context(df, "What is the recent momentum?")
```
