"""Example: Composable feature pipeline with FeatureSet."""

import alphakit as ak

# Load data
df = ak.load("MSFT", start="2024-01-01")

# Build a reusable feature pipeline
feature_set = ak.FeatureSet([
    ak.features.RSI(period=14),
    ak.features.MACD(),
    ak.features.BollingerBands(period=20, std=2.0),
    ak.features.Returns(periods=[1, 5, 21]),
    ak.features.RollingStats(windows=[5, 21], stats=["mean", "std"]),
    ak.features.Lags(columns=["close"], lags=[1, 2, 3, 5]),
])

# Apply features
df = feature_set.transform(df)
print(f"Features: {df.width} columns from {df.height} rows")
print(df.tail(5))

# Save the pipeline for reproducibility
feature_set.save("my_features.json")
print("\nSaved feature pipeline to my_features.json")

# Load it back and verify
loaded = ak.FeatureSet.load("my_features.json")
print(f"Loaded pipeline with {len(loaded)} steps: {loaded}")
