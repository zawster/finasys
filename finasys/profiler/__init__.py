"""finasys.profiler -- Smart financial data profiling.

One-call data profiling that answers: "Is this data ready for ML?"

Usage:
    import finasys as fs

    df = fs.load("AAPL", start="2024-01-01")
    report = fs.profiler.profile(df)
    print(fs.profiler.profile_summary(df))
"""

from __future__ import annotations

import numpy as np
import polars as pl

from finasys.profiler.report import (
    ColumnProfile,
    DataQualityReport,
    DistributionReport,
    ProfileReport,
)
from finasys.stats._utils import kurtosis as _kurtosis_np
from finasys.stats._utils import skewness as _skewness_np

__all__ = [
    "profile",
    "profile_summary",
    "ProfileReport",
    "ColumnProfile",
    "DataQualityReport",
    "DistributionReport",
]


def profile(
    df: pl.DataFrame,
    column: str = "close",
) -> ProfileReport:
    """Generate a comprehensive profile of financial data.

    Analyzes column statistics, data quality (gaps, outliers, splits),
    and return distribution properties in a single call.

    Args:
        df: DataFrame with financial data.
        column: Primary price column for distribution analysis.

    Returns:
        ProfileReport with all analysis results.
    """
    shape = (df.height, df.width)

    date_range = ("", "")
    if "timestamp" in df.columns and df.height > 0:
        ts = df["timestamp"]
        date_range = (str(ts.min()), str(ts.max()))

    symbols: list[str] = []
    if "symbol" in df.columns:
        symbols = df["symbol"].unique().sort().to_list()

    return ProfileReport(
        shape=shape,
        date_range=date_range,
        symbols=symbols,
        column_stats=_compute_column_profiles(df),
        quality=_compute_quality(df, column),
        distribution=_compute_distribution(df, column),
    )


def profile_summary(
    df: pl.DataFrame,
    column: str = "close",
) -> str:
    """Generate a text summary of the profile for LLM consumption.

    Args:
        df: DataFrame with financial data.
        column: Primary price column.

    Returns:
        Human-readable profile summary string.
    """
    report = profile(df, column)

    lines = []
    lines.append(f"DATA PROFILE | {report.shape[0]} rows x {report.shape[1]} columns")
    lines.append(f"Date range: {report.date_range[0]} to {report.date_range[1]}")
    if report.symbols:
        lines.append(f"Symbols: {', '.join(report.symbols)}")

    q = report.quality
    quality_issues = []
    if q.missing_dates:
        quality_issues.append(f"{len(q.missing_dates)} missing dates")
    if q.duplicate_rows > 0:
        quality_issues.append(f"{q.duplicate_rows} duplicate rows")
    if q.zero_volume_days > 0:
        quality_issues.append(f"{q.zero_volume_days} zero-volume days")
    if q.suspected_splits:
        quality_issues.append(f"{len(q.suspected_splits)} suspected splits")
    total_outliers = sum(q.price_outliers.values())
    if total_outliers > 0:
        quality_issues.append(f"{total_outliers} price outliers")

    if quality_issues:
        lines.append(f"Quality issues: {'; '.join(quality_issues)}")
    else:
        lines.append("Quality: No issues detected")

    d = report.distribution
    lines.append(
        f"Returns distribution: skew={d.returns_skewness:.3f}, "
        f"kurtosis={d.returns_kurtosis:.3f}, "
        f"{'normal' if d.is_normal else 'non-normal'} (JB p={d.jarque_bera_pvalue:.4f})"
    )
    lines.append(f"Tail ratio: {d.tail_ratio:.3f}")

    if column in report.column_stats:
        cs = report.column_stats[column]
        lines.append(
            f"{column}: mean={cs.mean:.2f}, std={cs.std:.2f}, "
            f"range=[{cs.min:.2f}, {cs.max:.2f}], "
            f"nulls={cs.null_count} ({cs.null_pct:.1f}%)"
        )

    return "\n".join(lines)


def _compute_column_profiles(df: pl.DataFrame) -> dict[str, ColumnProfile]:
    """Compute statistical profiles for all numeric columns."""
    profiles: dict[str, ColumnProfile] = {}

    for col_name in df.columns:
        col = df[col_name]
        dtype_str = str(col.dtype)
        count = col.len()
        null_count = col.null_count()
        null_pct = (null_count / count * 100) if count > 0 else 0.0

        cp = ColumnProfile(
            name=col_name,
            dtype=dtype_str,
            count=count,
            null_count=null_count,
            null_pct=null_pct,
        )

        if col.dtype.is_numeric():
            cp.mean = col.mean()
            cp.std = col.std()
            cp.min = col.min()
            cp.max = col.max()

            try:
                cp.skewness = col.skew()
            except Exception:
                cp.skewness = None

            try:
                arr = col.drop_nulls().to_numpy()
                if len(arr) > 3:
                    cp.kurtosis = _kurtosis_np(arr)
            except Exception:
                cp.kurtosis = None

            try:
                quantile_keys = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
                arr = col.drop_nulls().to_numpy()
                if len(arr) > 0:
                    quantile_vals = np.quantile(arr, quantile_keys)
                    for q, v in zip(quantile_keys, quantile_vals):
                        cp.quantiles[str(q)] = float(v)
            except Exception:
                pass

        profiles[col_name] = cp

    return profiles


def _compute_quality(df: pl.DataFrame, column: str) -> DataQualityReport:
    """Compute data quality metrics."""
    report = DataQualityReport()

    if "timestamp" in df.columns and df.height > 0:
        if "symbol" in df.columns:
            first_sym = df["symbol"].unique().sort().to_list()[0]
            sym_df = df.filter(pl.col("symbol") == first_sym)
            report.missing_dates = _find_missing_dates(sym_df)
        else:
            report.missing_dates = _find_missing_dates(df)

    report.duplicate_rows = int(df.is_duplicated().sum())

    if "volume" in df.columns:
        report.zero_volume_days = df.filter(pl.col("volume") == 0).height

    price_cols = [c for c in ["open", "high", "low", "close"] if c in df.columns]
    for pc in price_cols:
        col = df[pc]
        if col.dtype.is_numeric() and col.len() > 1:
            pct_change = (col / col.shift(1) - 1).drop_nulls()
            if pct_change.len() > 0:
                mean = pct_change.mean()
                std = pct_change.std()
                if std is not None and std > 1e-15:
                    outlier_count = pct_change.filter(
                        (pct_change - mean).abs() > 4 * std
                    ).len()
                    if outlier_count > 0:
                        report.price_outliers[pc] = outlier_count

    if column in df.columns:
        col = df[column]
        if col.dtype.is_numeric() and col.len() > 1:
            pct_change = col / col.shift(1) - 1
            if "timestamp" in df.columns:
                split_df = df.with_columns(pct_change.alias("_pct")).filter(
                    pl.col("_pct").abs() > 0.2
                )
                if split_df.height > 0:
                    report.suspected_splits = [
                        str(d) for d in split_df["timestamp"].to_list()
                    ]

    return report


def _find_missing_dates(df: pl.DataFrame) -> list[str]:
    """Find missing business dates in a DataFrame."""
    if df.height < 2:
        return []

    ts = df["timestamp"].sort()
    start = ts.min()
    end = ts.max()

    if start is None or end is None:
        return []

    # Generate all business days in range using numpy, then set-diff
    all_days = np.arange(
        np.datetime64(start),
        np.datetime64(end) + np.timedelta64(1, "D"),
        dtype="datetime64[D]",
    )
    business_days = all_days[np.is_busday(all_days)]
    existing = set(ts.to_list())
    from datetime import date as dt_date

    missing = []
    for bd in business_days:
        d = bd.astype("datetime64[D]").astype(dt_date)
        if d not in existing:
            missing.append(str(d))

    return missing


def _compute_distribution(df: pl.DataFrame, column: str) -> DistributionReport:
    """Compute return distribution characteristics."""
    report = DistributionReport()

    if column not in df.columns:
        return report

    col = df[column]
    if not col.dtype.is_numeric() or col.len() < 5:
        return report

    rets = (col / col.shift(1) - 1).drop_nulls().to_numpy()
    if len(rets) < 4:
        return report

    m2 = np.mean((rets - rets.mean()) ** 2)
    if m2 < 1e-15:
        return report

    skew = _skewness_np(rets)
    kurt = _kurtosis_np(rets)
    report.returns_skewness = skew
    report.returns_kurtosis = kurt

    # JB = n/6 * (S^2 + K^2/4), p-value via chi2(2) survival: e^(-x/2)
    n = len(rets)
    jb_stat = n / 6.0 * (skew**2 + kurt**2 / 4.0)
    report.jarque_bera_stat = jb_stat
    report.jarque_bera_pvalue = float(np.exp(-jb_stat / 2.0))
    report.is_normal = report.jarque_bera_pvalue > 0.05

    right_tail = np.quantile(rets, 0.95)
    left_tail = np.quantile(rets, 0.05)
    if abs(left_tail) > 1e-15:
        report.tail_ratio = float(abs(right_tail / left_tail))

    return report
