# Quality API

Reusable financial data quality checks. These complement `fs.profiler` by exposing checks directly.

## `fs.quality.detect_gaps(df, freq="1bd")`

Finds missing business dates. For multi-symbol DataFrames, returns a mapping of symbol to missing dates.

## `fs.quality.flag_outliers(df, method="zscore", threshold=4.0, columns=None)`

Appends boolean outlier columns such as `close_outlier`.

Supported methods:

- `"zscore"`
- `"iqr"`

## `fs.quality.detect_splits(df, threshold=0.3, column="close")`

Appends `suspected_split`, a boolean flag for large price jumps.

## `fs.quality.completeness_report(df)`

Returns a JSON-serializable summary with row/column counts, gaps, null counts, duplicate rows, zero-volume days, outlier counts, and suspected split counts.
