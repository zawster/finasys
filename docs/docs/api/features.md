# Features API

All feature functions take a Polars DataFrame and return a new DataFrame with columns appended. They never mutate the input.

All functions are **symbol-aware**: in multi-symbol DataFrames, calculations are done within each symbol (no cross-contamination).

---

## Technical Indicators

### `fs.features.sma(df, period=20, column="close")`
Simple Moving Average.

### `fs.features.ema(df, period=20, column="close")`
Exponential Moving Average.

### `fs.features.rsi(df, period=14, column="close")`
Relative Strength Index. Values: 0-100. Above 70 = overbought, below 30 = oversold.

### `fs.features.macd(df, fast=12, slow=26, signal=9, column="close")`
Moving Average Convergence Divergence. Adds columns: `macd_line`, `macd_signal`, `macd_hist`.

### `fs.features.bollinger(df, period=20, std=2.0, column="close")`
Bollinger Bands. Adds columns: `bb_middle`, `bb_upper`, `bb_lower`.

### `fs.features.atr(df, period=14)`
Average True Range. Measures volatility. Requires `high`, `low`, `close` columns.

### `fs.features.vwap(df)`
Volume Weighted Average Price. Requires `high`, `low`, `close`, `volume` columns.

### `fs.features.obv(df)`
On-Balance Volume. Requires `close` and `volume` columns.

### `fs.features.stochastic(df, k_period=14, d_period=3)`
Stochastic Oscillator. Adds columns: `stoch_k`, `stoch_d`. Values: 0-100.

### `fs.features.adx(df, period=14)`
Average Directional Index. Adds columns: `adx_14`, `plus_di`, `minus_di`.

### `fs.features.cci(df, period=20)`
Commodity Channel Index.

### `fs.features.williams_r(df, period=14)`
Williams %R. Values: -100 to 0.

### `fs.features.mfi(df, period=14)`
Money Flow Index. Volume-weighted RSI.

### `fs.features.roc(df, period=10, column="close")`
Rate of Change. Percentage change from `period` bars ago.

### `fs.features.momentum(df, period=10, column="close")`
Momentum. Price difference from `period` bars ago.

---

## Returns

### `fs.features.returns(df, periods=1, column="close")`
Simple percentage returns. `periods` can be `int` or `list[int]`.

```python
df = fs.features.returns(df, periods=[1, 5, 21])
# Adds: returns_1d, returns_5d, returns_21d
```

### `fs.features.log_returns(df, periods=1, column="close")`
Log returns (additive over time).

### `fs.features.cumulative_returns(df, column="close")`
Cumulative returns from the first data point.

### `fs.features.drawdown(df, column="close")`
Drawdown from running maximum. Adds: `drawdown`, `max_drawdown`.

---

## Rolling Statistics

### `fs.features.rolling_stats(df, windows=21, column="close", stats=None)`

```python
df = fs.features.rolling_stats(df, windows=[5, 21], stats=["mean", "std", "zscore"])
```

Available stats: `"mean"`, `"std"`, `"min"`, `"max"`, `"skew"`, `"zscore"`

---

## Lag Features

### `fs.features.lags(df, columns="close", lags=1)`

```python
df = fs.features.lags(df, columns=["close", "volume"], lags=[1, 3, 5])
# Adds: close_lag_1, close_lag_3, close_lag_5, volume_lag_1, ...
```

Only positive lags are allowed (look-ahead bias protection).

### `fs.features.validate_no_lookahead(df_full, df_partial, feature_columns)`
Validates that features don't use future data.

---

## Calendar Features

### `fs.features.calendar_features(df, column="timestamp")`
Adds: `day_of_week`, `month`, `quarter`, `week_of_year`, `is_month_start`, `is_month_end`, `is_quarter_end`.

---

## Cross-Sectional Features

For multi-symbol DataFrames. Ranks/scores across symbols at each timestamp.

### `fs.features.cross_rank(df, column="close")`
### `fs.features.cross_percentile(df, column="close")`
### `fs.features.cross_zscore(df, column="close")`

---

## Convenience

### `fs.features.add_all(df, indicators=True, returns_=True, lags_=None, rolling_windows=None, calendar=False)`

Add a standard set of features in one call.

```python
df = fs.features.add_all(df, lags_=[1, 5], rolling_windows=[5, 21], calendar=True)
```

---

## FeatureSet (Composable Pipeline)

```python
fs = fs.FeatureSet([
    fs.features.RSI(period=14),
    fs.features.MACD(),
    fs.features.BollingerBands(period=20),
    fs.features.ATR(period=14),
    fs.features.Returns(periods=[1, 5, 21]),
    fs.features.LogReturns(periods=1),
    fs.features.RollingStats(windows=[5, 21], stats=["mean", "std"]),
    fs.features.Lags(columns=["close"], lags=[1, 3, 5]),
    fs.features.Calendar(),
])

df = fs.transform(df)

# Save / load for reproducibility
fs.save("pipeline.json")
fs2 = fs.FeatureSet.load("pipeline.json")
```

Available step classes: `RSI`, `MACD`, `BollingerBands`, `ATR`, `Returns`, `LogReturns`, `RollingStats`, `Lags`, `Calendar`
