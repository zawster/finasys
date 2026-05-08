<p align="center">
  <img src="https://raw.githubusercontent.com/zawster/finasys/main/docs/docs/assets/logo_wide.png" alt="finasys" width="200">
</p>

<p align="center">
  <strong>From raw market data to ML-ready features in five lines of code.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/finasys/"><img src="https://img.shields.io/pypi/v/finasys.svg" alt="PyPI"></a>
  <a href="https://github.com/zawster/finasys/actions"><img src="https://github.com/zawster/finasys/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://codecov.io/gh/zawster/finasys"><img src="https://codecov.io/gh/zawster/finasys/branch/main/graph/badge.svg" alt="Coverage"></a>
  <a href="https://github.com/zawster/finasys/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python"></a>
</p>

**Documentation:** [finasys Docs](https://zawster.github.io/finasys)

---

finasys is a toolkit for *financial data processing — not manual wrangling — for ML pipelines and AI agents*. It lets you go from **raw market data to production-ready features** in a few lines of code, whether you're building trading models, running portfolio analysis, or powering financial AI agents.

finasys is **Polars-first** — every indicator and feature runs as a native Polars expression, making it 10-100x faster than pandas-based alternatives with **zero C dependencies** (no ta-lib build headaches). It supports **37+ international markets**, crypto, forex, commodities, and macro indicators out of the box. Learn more via our [official documentation](https://zawster.github.io/finasys) or start contributing via this GitHub repo.

## Quick Start

```python
import finasys as fs

# Load stock data (auto-cached with DuckDB)
df = fs.load("AAPL", start="2024-01-01")

# Add technical indicators + returns in one call
df = fs.features.add_all(df)

# Generate an LLM-ready summary
print(fs.agents.summarize(df))
```

## Install

```bash
pip install finasys
```

Optional extras:
```bash
pip install finasys[langchain]   # LangChain tool integration
pip install finasys[pandas]      # Pandas interop
pip install finasys[all]         # Everything
```

## Features

### Data Sources (`fs.load()`)
- Single `fs.load()` entry point for Yahoo Finance, CSV, and Parquet files
- Standardized OHLCV column names across all sources
- DuckDB-backed local caching (second call is instant)
- Multi-symbol fetching with automatic alignment

```python
df = fs.load("AAPL", start="2024-01-01")
df = fs.load(["AAPL", "GOOGL", "MSFT"], start="2024-01-01")
df = fs.load("./data/prices.csv")
```

### Feature Engineering (`fs.features`)
- 15+ technical indicators: RSI, MACD, Bollinger Bands, ATR, VWAP, OBV, Stochastic, ADX, CCI, Williams %R, MFI, ROC, Momentum
- Returns: simple, log, cumulative, drawdown
- Rolling statistics: mean, std, min, max, skew, z-score
- Lag features with built-in look-ahead bias protection
- Calendar features: day of week, month, quarter
- Cross-sectional: rank, percentile, z-score across symbols

All implemented in **pure Polars expressions** -- no ta-lib C dependency, 10-100x faster than pandas-ta.

```python
df = fs.features.rsi(df, period=14)
df = fs.features.macd(df)
df = fs.features.returns(df, periods=[1, 5, 21])
```

### Target / Label Engineering (`fs.features`)
- Forward returns for regression targets
- Ternary classification labels (up/flat/down) with configurable thresholds
- **Triple-barrier labeling** (Lopez de Prado method) -- the gold standard for financial ML
- Volatility-adjusted labels that adapt to the current regime

```python
# Forward returns for regression
df = fs.features.forward_returns(df, periods=[1, 5])

# Classification labels
df = fs.features.classify_returns(df, period=5, thresholds=(-0.01, 0.01))

# Triple-barrier method
df = fs.features.triple_barrier_labels(df, profit_take=0.02, stop_loss=0.02, max_holding=10)

# Volatility-adjusted labels (adapts to regime)
df = fs.features.volatility_adjusted_labels(df, period=5, vol_multiplier=1.0)
```

### Distribution Features (`fs.features`)
- Rolling kurtosis, skewness, tail ratio -- capture fat-tail dynamics
- Rolling Jarque-Bera normality test
- Z-score of returns vs rolling distribution

```python
df = fs.features.rolling_kurtosis(df, window=30)
df = fs.features.rolling_skewness(df, window=30)
df = fs.features.tail_ratio(df, window=30)
df = fs.features.zscore_returns(df, window=30)
```

### Market Regime Features (`fs.features`)
- Volatility regimes from fast/slow rolling volatility
- Trend strength via rolling Hurst-style approximation
- Combined market states such as trending/high-volatility or ranging/low-volatility
- Breakout flags and strength scores

```python
df = fs.features.volatility_regime(df, fast_window=21, slow_window=63)
df = fs.features.trend_strength(df, window=63)
df = fs.features.market_state(df)
df = fs.features.breakout_detection(df, window=20)
```

### Risk & Performance Metrics (`fs.stats`)
- Sharpe, Sortino, Calmar ratios
- Value at Risk (historical, parametric, Cornish-Fisher)
- Conditional VaR (Expected Shortfall)
- CAPM alpha/beta, information ratio
- Max drawdown duration tracking
- **Dual mode**: scalar for reporting, rolling columns for ML features

```python
# Scalar metrics (whole-series)
sharpe = fs.stats.sharpe_ratio(df)                         # => 1.47
var = fs.stats.value_at_risk(df, confidence=0.95)           # => -0.0216
cvar = fs.stats.cvar(df, confidence=0.95)                   # => -0.0285

# Rolling metrics (ML features)
df = fs.stats.sharpe_ratio(df, window=63)                   # adds sharpe_63
df = fs.stats.value_at_risk(df, window=63)                  # adds var_63
```

### Portfolio Analytics (`fs.portfolio`)
- Correlation and covariance matrices for multi-symbol DataFrames
- Pairwise rolling correlation
- Weighted and equal-weight portfolio returns
- Minimum-variance portfolio weights

```python
df = fs.load(["AAPL", "GOOGL", "MSFT"], start="2024-01-01")
corr = fs.portfolio.correlation_matrix(df)
portfolio = fs.portfolio.equal_weight_returns(df)
weights = fs.portfolio.minimum_variance_weights(df)
```

### Data Quality Checks (`fs.quality`)
- Missing business-day gaps
- Outlier flags
- Suspected split flags
- Completeness reports with nulls, duplicates, zero-volume days, gaps, and flags

```python
gaps = fs.quality.detect_gaps(df)
df = fs.quality.flag_outliers(df)
df = fs.quality.detect_splits(df)
report = fs.quality.completeness_report(df)
```

### Smart Profiler (`fs.profiler`)
- One-call data quality assessment for financial time series
- Detects: missing dates, price outliers, suspected stock splits, zero-volume days
- Distribution analysis: skewness, kurtosis, Jarque-Bera normality test, tail ratio
- LLM-ready text summaries and JSON-serializable structured reports

```python
# Text summary (great for LLM system prompts)
print(fs.profiler.profile_summary(df))
# DATA PROFILE | 252 rows x 7 columns
# Quality issues: 9 missing dates; 11 price outliers
# Returns distribution: skew=0.501, kurtosis=3.647, non-normal (JB p=0.0000)

# Full structured report
report = fs.profiler.profile(df)
report.quality.missing_dates      # ['2024-01-15', '2024-02-19', ...]
report.distribution.is_normal     # False
report.to_dict()                  # JSON-serializable
```

### AI Agent Tools (`fs.agents`)
- LLM-ready summaries of financial DataFrames
- Tool definitions in OpenAI function-calling format
- Extended tools for risk reports, portfolio analysis, stock screening, quality checks, and profile summaries
- Context extraction for RAG-style usage
- Schema descriptions for system prompts
- LangChain integration (optional)

```python
summary = fs.agents.summarize(df)
tools = fs.agents.tools(symbols=["AAPL", "GOOGL"])

from finasys.agents.langchain import get_tools
lc_tools = get_tools(symbols=["AAPL"])
```

### Composable Pipelines (`fs.FeatureSet`)
Serializable, reproducible feature pipelines with 21 built-in step classes.

```python
pipeline = fs.FeatureSet([
    fs.features.RSI(period=14),
    fs.features.Returns(periods=[1, 5, 21]),
    fs.features.RollingStats(windows=[5, 21]),
    fs.features.RollingKurtosis(window=30),
    fs.features.VolatilityRegime(),
    fs.features.ForwardReturns(periods=[1, 5]),
    fs.features.TripleBarrier(profit_take=0.02, stop_loss=0.02),
])
df = pipeline.transform(df)
pipeline.save("pipeline.json")  # version control your feature engineering
```

## Why finasys?

| | finasys | pandas-ta | ta-lib |
|---|---|---|---|
| **Engine** | Polars (fast) | pandas (slow) | C library |
| **Install** | `pip install finasys` | `pip install pandas-ta` | Requires C build tools |
| **ML Targets** | Triple-barrier, vol-adjusted labels | None | None |
| **Risk Metrics** | Sharpe, VaR, CVaR, alpha/beta | None | None |
| **Data Profiling** | Financial-specific quality checks | None | None |
| **AI Agent support** | Built-in | None | None |
| **Caching** | DuckDB auto-cache | None | None |
| **Look-ahead protection** | Built-in | None | None |

## License

Apache-2.0
