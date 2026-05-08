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

## Target / Label Engineering

Functions for creating supervised ML targets. These use **forward-looking data** and must be dropped before inference.

### `fs.features.forward_returns(df, periods=1, column="close")`

Forward-looking returns via negative shift. The most common ML target in financial modeling.

```python
df = fs.features.forward_returns(df, periods=[1, 5, 21])
# Adds: fwd_return_1d, fwd_return_5d, fwd_return_21d
# Last N rows are null (no future data available)
```

### `fs.features.classify_returns(df, period=5, thresholds=(-0.01, 0.01), column="close")`

Classify forward returns into ternary labels for classification models.

- `-1`: down (forward return < lower threshold)
- `0`: flat (between thresholds)
- `1`: up (forward return > upper threshold)

```python
df = fs.features.classify_returns(df, period=5, thresholds=(-0.01, 0.01))
# Adds: label_5d (values: -1, 0, 1)
```

### `fs.features.triple_barrier_labels(df, profit_take=0.02, stop_loss=0.02, max_holding=10, column="close")`

Lopez de Prado triple-barrier labeling method -- the gold standard for financial ML labeling from *Advances in Financial Machine Learning*.

Three barriers race:

- **Upper**: price rises by `profit_take` fraction (label = 1)
- **Lower**: price falls by `stop_loss` fraction (label = -1)
- **Vertical**: `max_holding` bars elapse (label = sign of return)

```python
df = fs.features.triple_barrier_labels(df, profit_take=0.02, stop_loss=0.02, max_holding=10)
# Adds: tb_label (1/-1/0), tb_duration (bars held), tb_return (exit return)
```

### `fs.features.volatility_adjusted_labels(df, period=5, vol_window=21, vol_multiplier=1.0, column="close")`

Classify forward returns relative to rolling volatility. Thresholds adapt to the current regime -- more robust than fixed thresholds.

- `up`: forward return > `vol_multiplier * rolling_std`
- `down`: forward return < `-vol_multiplier * rolling_std`
- `flat`: otherwise

```python
df = fs.features.volatility_adjusted_labels(df, period=5, vol_multiplier=1.0)
# Adds: vol_label_5d (values: -1, 0, 1)
```

---

## Distribution Features

Rolling distribution metrics that capture fat tails, non-normality, and tail risk dynamics. Powerful ML features for regime detection and risk prediction.

### `fs.features.rolling_skewness(df, window=63, column="close")`

Rolling skewness of returns. Negative skew = heavier left tail (common in equities).

```python
df = fs.features.rolling_skewness(df, window=63)
# Adds: rolling_skew_63
```

### `fs.features.rolling_kurtosis(df, window=63, column="close")`

Rolling excess kurtosis of returns. Values > 0 indicate fat tails (leptokurtic). Financial returns typically have positive excess kurtosis.

```python
df = fs.features.rolling_kurtosis(df, window=63)
# Adds: rolling_kurtosis_63
```

### `fs.features.tail_ratio(df, window=63, percentile=0.05, column="close")`

Ratio of the right tail (95th percentile) to the absolute value of the left tail (5th percentile). Values > 1 indicate positive skew.

```python
df = fs.features.tail_ratio(df, window=63)
# Adds: tail_ratio_63
```

### `fs.features.rolling_jarque_bera(df, window=63, column="close")`

Rolling Jarque-Bera test statistic. High values indicate non-normal returns. Computed as `JB = n/6 * (S^2 + K^2/4)`.

```python
df = fs.features.rolling_jarque_bera(df, window=63)
# Adds: rolling_jb_63
```

### `fs.features.zscore_returns(df, window=63, column="close")`

Z-score of the current return relative to its rolling distribution. Detects unusually large or small moves.

```python
df = fs.features.zscore_returns(df, window=63)
# Adds: zscore_returns_63
```

---

## Market Regime Features

### `fs.features.volatility_regime(df, fast_window=21, slow_window=63, column="close")`

Adds `vol_ratio` and `vol_regime`. `vol_regime` is `1` when fast volatility is above slow volatility, otherwise `0`.

### `fs.features.trend_strength(df, window=63, column="close")`

Adds `hurst_{window}` and `trend_direction`. `trend_direction` is `1`, `0`, or `-1`.

### `fs.features.market_state(df, vol_window=21, trend_window=63, column="close")`

Combines volatility and trend into `market_state`, such as `trending_high_vol` or `ranging_low_vol`.

### `fs.features.breakout_detection(df, window=20, n_std=2.0, column="close")`

Adds `breakout_{window}` and `breakout_strength_{window}`.

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

Available step classes:

| Category | Steps |
|----------|-------|
| **Indicators** | `RSI`, `MACD`, `BollingerBands`, `ATR` |
| **Returns** | `Returns`, `LogReturns` |
| **Features** | `RollingStats`, `Lags`, `Calendar` |
| **Targets** | `ForwardReturns`, `ClassifyReturns`, `TripleBarrier`, `VolAdjustedLabels` |
| **Distributions** | `RollingSkewness`, `RollingKurtosis`, `TailRatio`, `ZscoreReturns` |
| **Regime** | `VolatilityRegime`, `TrendStrength`, `MarketState`, `BreakoutDetection` |

```python
# ML pipeline with targets and distribution features
pipeline = fs.FeatureSet([
    fs.features.RSI(period=14),
    fs.features.Returns(periods=[1, 5, 21]),
    fs.features.RollingKurtosis(window=30),
    fs.features.ZscoreReturns(window=30),
    fs.features.ForwardReturns(periods=[1, 5]),
    fs.features.ClassifyReturns(period=5),
    fs.features.TripleBarrier(profit_take=0.02, stop_loss=0.02, max_holding=10),
])

df = pipeline.transform(df)
pipeline.save("ml_pipeline.json")
```
