# Profiler API

One-call financial data profiling that answers: **"Is this data ready for ML?"**

The profiler analyzes column statistics, data quality issues specific to financial time series (gaps, splits, outliers), and return distribution properties.

---

## `fs.profiler.profile(df, column="close")`

Generate a comprehensive profile report for a financial DataFrame.

```python
import finasys as fs

df = fs.load("AAPL", start="2024-01-01")
report = fs.profiler.profile(df)

print(report.shape)          # (252, 7)
print(report.date_range)     # ('2024-01-02', '2024-12-31')
print(report.symbols)        # ['AAPL']
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | required | DataFrame with financial data |
| `column` | `str` | `"close"` | Primary price column for distribution analysis |

**Returns:** `ProfileReport` dataclass (see below).

---

## `fs.profiler.profile_summary(df, column="close")`

Generate a text summary designed for LLM consumption. Can be plugged directly into agent system prompts.

```python
summary = fs.profiler.profile_summary(df)
print(summary)
```

**Example output:**
```
DATA PROFILE | 252 rows x 7 columns
Date range: 2024-01-02 to 2024-12-31
Symbols: AAPL
Quality issues: 9 missing dates; 11 price outliers
Returns distribution: skew=0.501, kurtosis=3.647, non-normal (JB p=0.0000)
Tail ratio: 0.987
close: mean=205.65, std=25.58, range=[163.51, 257.61], nulls=0 (0.0%)
```

**Returns:** `str`

---

## Report Dataclasses

### `ProfileReport`

The top-level report containing all analysis results.

| Field | Type | Description |
|-------|------|-------------|
| `shape` | `tuple[int, int]` | `(rows, columns)` |
| `date_range` | `tuple[str, str]` | `(start_date, end_date)` |
| `symbols` | `list[str]` | Symbols found in the data |
| `column_stats` | `dict[str, ColumnProfile]` | Per-column statistics |
| `quality` | `DataQualityReport` | Data quality assessment |
| `distribution` | `DistributionReport` | Return distribution analysis |

```python
report = fs.profiler.profile(df)

# Serialize to dict (JSON-compatible)
data = report.to_dict()

import json
json.dumps(data, default=str)  # works
```

---

### `ColumnProfile`

Statistical profile of a single column. Computed for all numeric columns automatically.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Column name |
| `dtype` | `str` | Polars data type |
| `count` | `int` | Total row count |
| `null_count` | `int` | Number of null values |
| `null_pct` | `float` | Null percentage (0-100) |
| `mean` | `float` | Mean (numeric columns only) |
| `std` | `float` | Standard deviation |
| `min` | `float` | Minimum value |
| `max` | `float` | Maximum value |
| `skewness` | `float` | Skewness |
| `kurtosis` | `float` | Excess kurtosis |
| `quantiles` | `dict[str, float]` | Quantiles at 1%, 5%, 25%, 50%, 75%, 95%, 99% |

```python
cs = report.column_stats["close"]
print(f"Mean: {cs.mean:.2f}")
print(f"Std: {cs.std:.2f}")
print(f"Median: {cs.quantiles['0.5']:.2f}")
print(f"Nulls: {cs.null_count} ({cs.null_pct:.1f}%)")
```

---

### `DataQualityReport`

Financial-specific data quality checks. Detects issues that generic profilers miss.

| Field | Type | Description |
|-------|------|-------------|
| `missing_dates` | `list[str]` | Business days with no data (holidays, gaps) |
| `duplicate_rows` | `int` | Number of duplicate rows |
| `zero_volume_days` | `int` | Days with zero trading volume |
| `price_outliers` | `dict[str, int]` | Per-column count of >4-sigma daily moves |
| `suspected_splits` | `list[str]` | Dates with >20% overnight price changes |

```python
q = report.quality

# Check for data gaps
if q.missing_dates:
    print(f"Warning: {len(q.missing_dates)} missing trading dates")
    print(f"  First 5: {q.missing_dates[:5]}")

# Check for stock splits (unadjusted data)
if q.suspected_splits:
    print(f"Warning: {len(q.suspected_splits)} suspected stock splits")
    print("  Consider using adjusted close prices")

# Outlier check
for col, count in q.price_outliers.items():
    print(f"  {col}: {count} outliers (>4 sigma)")
```

---

### `DistributionReport`

Return distribution characteristics. Financial returns are famously non-normal -- this tells you how non-normal.

| Field | Type | Description |
|-------|------|-------------|
| `returns_skewness` | `float` | Skewness of daily returns (negative = left tail heavier) |
| `returns_kurtosis` | `float` | Excess kurtosis (>0 = fat tails, typical for equities) |
| `jarque_bera_stat` | `float` | Jarque-Bera test statistic |
| `jarque_bera_pvalue` | `float` | JB p-value (<0.05 = reject normality) |
| `is_normal` | `bool` | `True` if p-value > 0.05 |
| `tail_ratio` | `float` | Right tail / left tail ratio (>1 = positive skew) |

```python
d = report.distribution

if not d.is_normal:
    print("Returns are non-normal (typical for financial data)")
    print(f"  Kurtosis: {d.returns_kurtosis:.2f} (0 = normal, >0 = fat tails)")
    print(f"  Skewness: {d.returns_skewness:.2f}")
```

!!! info "Jarque-Bera test"
    The JB test checks whether returns follow a normal distribution by examining skewness and kurtosis. Formula: `JB = n/6 * (S^2 + K^2/4)`. For financial returns, normality is almost always rejected -- this is expected and important to know when choosing models.

---

## Multi-Symbol Profiling

The profiler works with multi-symbol DataFrames:

```python
df = fs.load(["AAPL", "GOOGL", "MSFT"], start="2024-01-01")
report = fs.profiler.profile(df)

print(report.symbols)  # ['AAPL', 'GOOGL', 'MSFT']
print(fs.profiler.profile_summary(df))
```

Column statistics are computed across the entire DataFrame. Date gap detection runs per-symbol to avoid false positives from different trading calendars.
