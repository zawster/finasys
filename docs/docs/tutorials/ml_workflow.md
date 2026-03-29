# Building an ML-Ready Financial Dataset

This tutorial walks through the complete workflow: **profile data, engineer features, create targets, and assess risk** -- everything you need before training a financial ML model.

## Step 1: Load and Profile

Always start by profiling your data to catch quality issues before they corrupt your model.

```python
import finasys as fs

df = fs.load("AAPL", start="2024-01-01")

# One-call profiling
print(fs.profiler.profile_summary(df))
```

```
DATA PROFILE | 252 rows x 7 columns
Date range: 2024-01-02 to 2024-12-31
Symbols: AAPL
Quality issues: 9 missing dates; 11 price outliers
Returns distribution: skew=0.501, kurtosis=3.647, non-normal (JB p=0.0000)
Tail ratio: 0.987
close: mean=205.65, std=25.58, range=[163.51, 257.61], nulls=0 (0.0%)
```

The profiler tells us: 9 missing dates (US holidays -- expected), 11 price outliers (>4-sigma moves), and returns are non-normal with fat tails (kurtosis=3.65) -- all typical for equity data.

For a deeper look:

```python
report = fs.profiler.profile(df)

# Check for data issues
if report.quality.suspected_splits:
    print("Warning: suspected stock splits detected!")
if report.quality.zero_volume_days > 0:
    print("Warning: zero-volume days found")

# Distribution details
d = report.distribution
print(f"Skewness: {d.returns_skewness:.3f}")
print(f"Kurtosis: {d.returns_kurtosis:.3f}")
print(f"Normal? {d.is_normal}")  # Almost always False for financial data
```

## Step 2: Engineer Features

Build features using both the functional API and the composable pipeline.

```python
# Technical indicators
df = fs.features.add_all(df, indicators=True, returns_=True, rolling_windows=[5, 21])

# Distribution features -- capture fat tail dynamics
df = fs.features.rolling_kurtosis(df, window=30)
df = fs.features.rolling_skewness(df, window=30)
df = fs.features.zscore_returns(df, window=30)
df = fs.features.tail_ratio(df, window=30)

print(f"Features: {df.shape[1]} columns")
```

Or use a `FeatureSet` for a reproducible, serializable pipeline:

```python
pipeline = fs.FeatureSet([
    fs.features.RSI(period=14),
    fs.features.Returns(periods=[1, 5, 21]),
    fs.features.RollingStats(windows=[5, 21]),
    fs.features.Lags(columns=["close"], lags=[1, 3, 5]),
    fs.features.RollingKurtosis(window=30),
    fs.features.ZscoreReturns(window=30),
])

df = pipeline.transform(df)
pipeline.save("my_features.json")  # version-controlled reproducibility
```

## Step 3: Create Targets

Choose a labeling method based on your model type.

### For regression: Forward Returns

```python
df = fs.features.forward_returns(df, periods=[1, 5])
# Predicting next-day and next-week returns
```

### For classification: Fixed Thresholds

```python
df = fs.features.classify_returns(df, period=5, thresholds=(-0.01, 0.01))
# Labels: -1 (down >1%), 0 (flat), 1 (up >1%)
```

### For classification: Volatility-Adjusted Thresholds

Better than fixed thresholds because they adapt to the current volatility regime:

```python
df = fs.features.volatility_adjusted_labels(df, period=5, vol_multiplier=1.0)
# Thresholds = +/- 1x rolling standard deviation
```

### For advanced models: Triple Barrier

The gold standard from Lopez de Prado's *Advances in Financial Machine Learning*:

```python
df = fs.features.triple_barrier_labels(
    df,
    profit_take=0.02,    # +2% take profit
    stop_loss=0.02,      # -2% stop loss
    max_holding=10,      # 10-day maximum hold
)
# tb_label: which barrier was hit first (1=profit, -1=stop, 0=expiry)
# tb_duration: how many bars until exit
# tb_return: actual return at exit
```

## Step 4: Assess Risk

Before committing to a strategy, evaluate risk metrics:

```python
# Overall metrics
print(f"Sharpe:  {fs.stats.sharpe_ratio(df):.2f}")
print(f"Sortino: {fs.stats.sortino_ratio(df):.2f}")
print(f"Calmar:  {fs.stats.calmar_ratio(df):.2f}")
print(f"VaR 95%: {fs.stats.value_at_risk(df, confidence=0.95):.4%}")
print(f"CVaR 95%: {fs.stats.cvar(df, confidence=0.95):.4%}")

# Drawdown analysis
df = fs.stats.max_drawdown_duration(df)
max_dd = df["dd_max_duration"].max()
print(f"Longest drawdown: {max_dd} trading days")
```

Add rolling risk metrics as features:

```python
df = fs.stats.sharpe_ratio(df, window=63)     # Rolling 3-month Sharpe
df = fs.stats.value_at_risk(df, window=63)     # Rolling VaR
df = fs.stats.sortino_ratio(df, window=63)     # Rolling Sortino
```

## Step 5: Prepare for Training

Separate features from targets and drop rows with null values from warm-up periods:

```python
import polars as pl

# Define feature and target columns
target_cols = ["fwd_return_1d", "fwd_return_5d", "label_5d",
               "tb_label", "tb_duration", "tb_return", "vol_label_5d"]
meta_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
feature_cols = [c for c in df.columns if c not in target_cols + meta_cols]

# Drop warm-up nulls
df_clean = df.drop_nulls(subset=feature_cols)
print(f"Training rows: {df_clean.shape[0]} (dropped {df.shape[0] - df_clean.shape[0]} warm-up rows)")
print(f"Features: {len(feature_cols)}")
print(f"Targets: {[c for c in target_cols if c in df.columns]}")
```

## Multi-Symbol Workflow

Everything works seamlessly with multiple symbols:

```python
df = fs.load(["AAPL", "GOOGL", "MSFT"], start="2024-01-01")

# Profile all symbols
print(fs.profiler.profile_summary(df))

# Features are computed per-symbol (no cross-contamination)
df = fs.features.add_all(df)
df = fs.features.forward_returns(df, periods=[1, 5])
df = fs.features.triple_barrier_labels(df)

# Cross-sectional features (rank across symbols at each timestamp)
df = fs.features.cross_rank(df, column="close")
df = fs.features.cross_zscore(df, column="close")

# Risk per symbol
for sym in ["AAPL", "GOOGL", "MSFT"]:
    sym_df = df.filter(pl.col("symbol") == sym)
    sharpe = fs.stats.sharpe_ratio(sym_df)
    print(f"{sym}: Sharpe = {sharpe:.2f}")
```

## Complete Pipeline Example

Putting it all together in a serializable pipeline:

```python
import finasys as fs

# Load
df = fs.load("AAPL", start="2023-01-01")

# Profile
print(fs.profiler.profile_summary(df))

# Feature + target pipeline
pipeline = fs.FeatureSet([
    # Technical features
    fs.features.RSI(period=14),
    fs.features.BollingerBands(period=20),
    fs.features.Returns(periods=[1, 5, 21]),
    fs.features.RollingStats(windows=[5, 21]),
    fs.features.Lags(columns=["close"], lags=[1, 3, 5]),
    # Distribution features
    fs.features.RollingKurtosis(window=30),
    fs.features.ZscoreReturns(window=30),
    # Targets
    fs.features.ForwardReturns(periods=[1, 5]),
    fs.features.TripleBarrier(profit_take=0.02, stop_loss=0.02),
])

df = pipeline.transform(df)
pipeline.save("production_pipeline.json")

# Risk check
print(f"Sharpe: {fs.stats.sharpe_ratio(df):.2f}")
print(f"VaR 95%: {fs.stats.value_at_risk(df):.4%}")
print(f"Max DD duration: {fs.stats.max_drawdown_duration(df)['dd_max_duration'].max()} days")

# ML-ready
df_clean = df.drop_nulls()
print(f"Ready: {df_clean.shape[0]} rows x {df_clean.shape[1]} columns")
```
