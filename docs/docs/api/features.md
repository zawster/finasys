# Features API

All feature functions take a Polars DataFrame and return a new DataFrame with columns appended. They never mutate the input.

All functions are **symbol-aware**: in multi-symbol DataFrames, calculations are done within each symbol (no cross-contamination).

---

## Technical Indicators

### `ak.features.sma(df, period=20, column="close")`
Simple Moving Average.

### `ak.features.ema(df, period=20, column="close")`
Exponential Moving Average.

### `ak.features.rsi(df, period=14, column="close")`
Relative Strength Index. Values: 0-100. Above 70 = overbought, below 30 = oversold.

### `ak.features.macd(df, fast=12, slow=26, signal=9, column="close")`
Moving Average Convergence Divergence. Adds columns: `macd_line`, `macd_signal`, `macd_hist`.

### `ak.features.bollinger(df, period=20, std=2.0, column="close")`
Bollinger Bands. Adds columns: `bb_middle`, `bb_upper`, `bb_lower`.

### `ak.features.atr(df, period=14)`
Average True Range. Measures volatility. Requires `high`, `low`, `close` columns.

### `ak.features.vwap(df)`
Volume Weighted Average Price. Requires `high`, `low`, `close`, `volume` columns.

### `ak.features.obv(df)`
On-Balance Volume. Requires `close` and `volume` columns.

### `ak.features.stochastic(df, k_period=14, d_period=3)`
Stochastic Oscillator. Adds columns: `stoch_k`, `stoch_d`. Values: 0-100.

### `ak.features.adx(df, period=14)`
Average Directional Index. Adds columns: `adx_14`, `plus_di`, `minus_di`.

### `ak.features.cci(df, period=20)`
Commodity Channel Index.

### `ak.features.williams_r(df, period=14)`
Williams %R. Values: -100 to 0.

### `ak.features.mfi(df, period=14)`
Money Flow Index. Volume-weighted RSI.

### `ak.features.roc(df, period=10, column="close")`
Rate of Change. Percentage change from `period` bars ago.

### `ak.features.momentum(df, period=10, column="close")`
Momentum. Price difference from `period` bars ago.

---

## Returns

### `ak.features.returns(df, periods=1, column="close")`
Simple percentage returns. `periods` can be `int` or `list[int]`.

```python
df = ak.features.returns(df, periods=[1, 5, 21])
# Adds: returns_1d, returns_5d, returns_21d
```

### `ak.features.log_returns(df, periods=1, column="close")`
Log returns (additive over time).

### `ak.features.cumulative_returns(df, column="close")`
Cumulative returns from the first data point.

### `ak.features.drawdown(df, column="close")`
Drawdown from running maximum. Adds: `drawdown`, `max_drawdown`.

---

## Rolling Statistics

### `ak.features.rolling_stats(df, windows=21, column="close", stats=None)`

```python
df = ak.features.rolling_stats(df, windows=[5, 21], stats=["mean", "std", "zscore"])
```

Available stats: `"mean"`, `"std"`, `"min"`, `"max"`, `"skew"`, `"zscore"`

---

## Lag Features

### `ak.features.lags(df, columns="close", lags=1)`

```python
df = ak.features.lags(df, columns=["close", "volume"], lags=[1, 3, 5])
# Adds: close_lag_1, close_lag_3, close_lag_5, volume_lag_1, ...
```

Only positive lags are allowed (look-ahead bias protection).

### `ak.features.validate_no_lookahead(df_full, df_partial, feature_columns)`
Validates that features don't use future data.

---

## Calendar Features

### `ak.features.calendar_features(df, column="timestamp")`
Adds: `day_of_week`, `month`, `quarter`, `week_of_year`, `is_month_start`, `is_month_end`, `is_quarter_end`.

---

## Cross-Sectional Features

For multi-symbol DataFrames. Ranks/scores across symbols at each timestamp.

### `ak.features.cross_rank(df, column="close")`
### `ak.features.cross_percentile(df, column="close")`
### `ak.features.cross_zscore(df, column="close")`

---

## Convenience

### `ak.features.add_all(df, indicators=True, returns_=True, lags_=None, rolling_windows=None, calendar=False)`

Add a standard set of features in one call.

```python
df = ak.features.add_all(df, lags_=[1, 5], rolling_windows=[5, 21], calendar=True)
```

---

## FeatureSet (Composable Pipeline)

```python
fs = ak.FeatureSet([
    ak.features.RSI(period=14),
    ak.features.MACD(),
    ak.features.BollingerBands(period=20),
    ak.features.ATR(period=14),
    ak.features.Returns(periods=[1, 5, 21]),
    ak.features.LogReturns(periods=1),
    ak.features.RollingStats(windows=[5, 21], stats=["mean", "std"]),
    ak.features.Lags(columns=["close"], lags=[1, 3, 5]),
    ak.features.Calendar(),
])

df = fs.transform(df)

# Save / load for reproducibility
fs.save("pipeline.json")
fs2 = ak.FeatureSet.load("pipeline.json")
```

Available step classes: `RSI`, `MACD`, `BollingerBands`, `ATR`, `Returns`, `LogReturns`, `RollingStats`, `Lags`, `Calendar`
