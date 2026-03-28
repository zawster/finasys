# Stats API

Risk and performance metrics for financial time series. Every function supports two modes:

- **Scalar mode** (`window=None`): returns a single `float` for the entire series -- ideal for reporting, dashboards, and agent responses.
- **Rolling mode** (`window=N`): appends a rolling column to the DataFrame -- ideal for ML features and time-varying analysis.

All functions are **symbol-aware** in multi-symbol DataFrames.

---

## Risk Metrics

### `fs.stats.sharpe_ratio(df, window=None, risk_free_rate=0.0, column="close")`

Annualized Sharpe ratio: excess return per unit of total risk.

```python
import finasys as fs

df = fs.load("AAPL", start="2024-01-01")

# Scalar -- one number for the whole series
sharpe = fs.stats.sharpe_ratio(df)                     # => 1.47
sharpe = fs.stats.sharpe_ratio(df, risk_free_rate=0.05) # => 1.25

# Rolling -- column appended to DataFrame
df = fs.stats.sharpe_ratio(df, window=63)
# Adds: sharpe_63
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | required | DataFrame with price data |
| `window` | `int` or `None` | `None` | Rolling window size. `None` = scalar result |
| `risk_free_rate` | `float` | `0.0` | Annual risk-free rate (e.g., `0.05` for 5%) |
| `column` | `str` | `"close"` | Price column name |

**Returns:** `float` (scalar mode) or `pl.DataFrame` with `sharpe_{window}` column (rolling mode)

---

### `fs.stats.sortino_ratio(df, window=None, risk_free_rate=0.0, column="close")`

Annualized Sortino ratio: excess return per unit of **downside** risk. Unlike Sharpe, only penalizes negative volatility -- better for assets with asymmetric return distributions.

```python
sortino = fs.stats.sortino_ratio(df)  # => 2.35

# Rolling
df = fs.stats.sortino_ratio(df, window=63)
# Adds: sortino_63
```

**Parameters:** Same as `sharpe_ratio`.

**Returns:** `float` or `pl.DataFrame` with `sortino_{window}` column.

---

### `fs.stats.calmar_ratio(df, column="close")`

Annualized return divided by maximum drawdown. A key hedge fund metric that measures return per unit of peak-to-trough risk.

```python
calmar = fs.stats.calmar_ratio(df)  # => 2.32
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | required | DataFrame with price data |
| `column` | `str` | `"close"` | Price column name |

**Returns:** `float`

---

### `fs.stats.value_at_risk(df, confidence=0.95, method="historical", window=None, column="close")`

Value at Risk: the maximum expected single-period loss at a given confidence level. Returns a negative number representing the loss threshold.

Three methods are available:

| Method | Description |
|--------|-------------|
| `"historical"` | Empirical quantile of actual returns (no distribution assumption) |
| `"parametric"` | Assumes normal distribution (uses scipy) |
| `"cornish_fisher"` | Adjusts for skewness and kurtosis (no scipy needed) |

```python
# 95% VaR -- "on 95% of days, the loss won't exceed this"
var = fs.stats.value_at_risk(df, confidence=0.95)                      # => -0.0216
var = fs.stats.value_at_risk(df, confidence=0.99, method="cornish_fisher")  # => -0.0341

# Rolling VaR
df = fs.stats.value_at_risk(df, confidence=0.95, window=63)
# Adds: var_63
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | required | DataFrame with price data |
| `confidence` | `float` | `0.95` | Confidence level (0.90, 0.95, 0.99) |
| `method` | `str` | `"historical"` | VaR method: `"historical"`, `"parametric"`, `"cornish_fisher"` |
| `window` | `int` or `None` | `None` | Rolling window size |
| `column` | `str` | `"close"` | Price column name |

**Returns:** `float` (negative, representing loss) or `pl.DataFrame` with `var_{window}` column.

---

### `fs.stats.cvar(df, confidence=0.95, window=None, column="close")`

Conditional VaR (Expected Shortfall): the expected loss **given** that the loss exceeds VaR. Always more severe than VaR -- it captures tail risk.

```python
cvar_val = fs.stats.cvar(df, confidence=0.95)  # => -0.0285

# CVaR is always worse than VaR
var = fs.stats.value_at_risk(df, confidence=0.95)
assert cvar_val <= var  # True

# Rolling
df = fs.stats.cvar(df, confidence=0.95, window=63)
# Adds: cvar_63
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | required | DataFrame with price data |
| `confidence` | `float` | `0.95` | Confidence level |
| `window` | `int` or `None` | `None` | Rolling window size |
| `column` | `str` | `"close"` | Price column name |

**Returns:** `float` or `pl.DataFrame` with `cvar_{window}` column.

---

### `fs.stats.max_drawdown_duration(df, column="close")`

Track how long the price stays below its previous peak. Extends the existing `drawdown()` function with duration information.

```python
df = fs.stats.max_drawdown_duration(df)
# Adds: dd_duration (bars in current drawdown, 0 at new high)
#        dd_max_duration (longest drawdown so far)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | required | DataFrame with price data |
| `column` | `str` | `"close"` | Price column name |

**Returns:** `pl.DataFrame` with `dd_duration` and `dd_max_duration` columns appended.

---

## Performance Metrics

### `fs.stats.alpha_beta(df, benchmark_col="benchmark_close", column="close", window=None)`

CAPM alpha and beta versus a benchmark. Beta measures market sensitivity; alpha measures excess return beyond what beta explains.

!!! note "Benchmark column required"
    Your DataFrame must include a benchmark price column (e.g., S&P 500 prices). Load it separately and join, or add it as a column.

```python
import polars as pl

# Load asset and benchmark
asset = fs.load("AAPL", start="2024-01-01")
bench = fs.load("^GSPC", start="2024-01-01")

# Join on timestamp
df = asset.join(
    bench.select(["timestamp", pl.col("close").alias("benchmark_close")]),
    on="timestamp",
)

# Scalar
result = fs.stats.alpha_beta(df, benchmark_col="benchmark_close")
print(result)  # {"alpha": 0.15, "beta": 1.23}

# Rolling
df = fs.stats.alpha_beta(df, benchmark_col="benchmark_close", window=63)
# Adds: alpha_63, beta_63
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | required | DataFrame with asset and benchmark prices |
| `benchmark_col` | `str` | `"benchmark_close"` | Column name for benchmark prices |
| `column` | `str` | `"close"` | Column name for asset prices |
| `window` | `int` or `None` | `None` | Rolling window size |

**Returns:** `dict[str, float]` with `"alpha"` and `"beta"` keys (scalar mode) or `pl.DataFrame` with `alpha_{window}` and `beta_{window}` columns.

---

### `fs.stats.information_ratio(df, benchmark_col="benchmark_close", column="close")`

Active return divided by tracking error. Measures how consistently the asset outperforms its benchmark.

```python
ir = fs.stats.information_ratio(df, benchmark_col="benchmark_close")
# => 0.85
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | required | DataFrame with asset and benchmark prices |
| `benchmark_col` | `str` | `"benchmark_close"` | Benchmark price column |
| `column` | `str` | `"close"` | Asset price column |

**Returns:** `float`

---

## Quick Reference

| Metric | What it measures | Higher = better? |
|--------|-----------------|------------------|
| **Sharpe** | Return per unit of total risk | Yes |
| **Sortino** | Return per unit of downside risk | Yes |
| **Calmar** | Return per unit of max drawdown | Yes |
| **VaR** | Maximum expected loss at confidence level | Less negative = better |
| **CVaR** | Expected loss in the tail | Less negative = better |
| **Alpha** | Excess return vs benchmark | Positive = outperforming |
| **Beta** | Market sensitivity | 1.0 = market-like |
| **Information Ratio** | Consistency of outperformance | Higher = more consistent |
