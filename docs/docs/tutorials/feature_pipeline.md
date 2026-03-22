# Building a Feature Pipeline

This tutorial shows how to build a reusable, serializable feature engineering pipeline.

## Step 1: Load Data

```python
import alphakit as ak

df = ak.load("AAPL", start="2024-01-01")
```

## Step 2: Build the Pipeline

```python
pipeline = ak.FeatureSet([
    ak.features.RSI(period=14),
    ak.features.MACD(fast=12, slow=26, signal=9),
    ak.features.BollingerBands(period=20),
    ak.features.ATR(period=14),
    ak.features.Returns(periods=[1, 5, 21]),
    ak.features.RollingStats(windows=[5, 21], stats=["mean", "std"]),
    ak.features.Lags(columns=["close"], lags=[1, 2, 3, 5]),
    ak.features.Calendar(),
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
loaded = ak.FeatureSet.load("my_pipeline.json")
new_df = loaded.transform(ak.load("MSFT", start="2024-01-01"))
```
