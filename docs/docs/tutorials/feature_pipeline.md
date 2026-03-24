# Building a Feature Pipeline

This tutorial shows how to build a reusable, serializable feature engineering pipeline.

## Step 1: Load Data

```python
import finasys as fs

df = fs.load("AAPL", start="2024-01-01")
```

## Step 2: Build the Pipeline

```python
pipeline = fs.FeatureSet([
    fs.features.RSI(period=14),
    fs.features.MACD(fast=12, slow=26, signal=9),
    fs.features.BollingerBands(period=20),
    fs.features.ATR(period=14),
    fs.features.Returns(periods=[1, 5, 21]),
    fs.features.RollingStats(windows=[5, 21], stats=["mean", "std"]),
    fs.features.Lags(columns=["close"], lags=[1, 2, 3, 5]),
    fs.features.Calendar(),
])
```

## Step 3: Transform

```python
df = pipeline.transform(df)
print(f"{df.width} features from {df.height} rows")
```

## Step 4: Save for Reproducibility

```python
pipeline.save("my_pipeline.json")

# Later, reload and apply to new data
loaded = fs.FeatureSet.load("my_pipeline.json")
new_df = loaded.transform(fs.load("MSFT", start="2024-01-01"))
```
