"""Distribution analysis features for financial time series.

Rolling distribution metrics capture fat tails, non-normality, and
tail risk dynamics -- powerful ML features for regime detection and
risk prediction.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from finasys.features.utils import has_multi_symbols, symbol_aware


def _apply_rolling_map(
    df: pl.DataFrame,
    window: int,
    column: str,
    map_fn,
    output_name: str,
) -> pl.DataFrame:
    """Apply a rolling map function over returns, handling multi-symbol dispatch.

    Args:
        df: Input DataFrame.
        window: Rolling window size.
        column: Price column to compute returns from.
        map_fn: Function(pl.Series) -> pl.Series that operates on a returns series.
        output_name: Name for the output column.
    """
    if has_multi_symbols(df):
        ret_col = f"_ret_{output_name}"
        df_tmp = df.with_columns((pl.col(column) / pl.col(column).shift(1).over("symbol") - 1).alias(ret_col))
        frames = []
        for sym in df_tmp["symbol"].unique().sort().to_list():
            sym_df = df_tmp.filter(pl.col("symbol") == sym)
            series = map_fn(sym_df[ret_col])
            sym_df = sym_df.with_columns(series.alias(output_name))
            frames.append(sym_df)
        return pl.concat(frames).sort(["timestamp", "symbol"]).drop(ret_col)
    else:
        ret_col = f"_ret_{output_name}"
        df_tmp = df.with_columns((pl.col(column) / pl.col(column).shift(1) - 1).alias(ret_col))
        series = map_fn(df_tmp[ret_col])
        return df_tmp.with_columns(series.alias(output_name)).drop(ret_col)


def rolling_skewness(
    df: pl.DataFrame,
    window: int = 63,
    column: str = "close",
) -> pl.DataFrame:
    """Rolling skewness of returns.

    Positive skew = more extreme positive returns.
    Negative skew = more extreme negative returns (common in equities).

    Args:
        df: DataFrame with price data.
        window: Rolling window size.
        column: Price column.

    Returns:
        DataFrame with rolling_skew_{window} column appended.
    """
    col = pl.col(column)
    ret = col / col.shift(1) - 1
    expr = ret.rolling_skew(window_size=window).alias(f"rolling_skew_{window}")
    return df.with_columns(symbol_aware(expr, df))


def rolling_kurtosis(
    df: pl.DataFrame,
    window: int = 63,
    column: str = "close",
) -> pl.DataFrame:
    """Rolling excess kurtosis of returns.

    Values > 0 indicate fat tails (leptokurtic).
    Computed via the fourth central moment: E[(X-mu)^4] / sigma^4 - 3.

    Args:
        df: DataFrame with price data.
        window: Rolling window size.
        column: Price column.

    Returns:
        DataFrame with rolling_kurtosis_{window} column appended.
    """
    from finasys.stats._utils import kurtosis as _kurtosis_np

    def _compute(s: pl.Series) -> pl.Series:
        arr = s.to_numpy()
        n = len(arr)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            chunk = arr[i - window + 1 : i + 1]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) < 4:
                continue
            result[i] = _kurtosis_np(valid)
        return pl.Series(result)

    return _apply_rolling_map(df, window, column, _compute, f"rolling_kurtosis_{window}")


def tail_ratio(
    df: pl.DataFrame,
    window: int = 63,
    percentile: float = 0.05,
    column: str = "close",
) -> pl.DataFrame:
    """Rolling tail ratio of returns.

    Ratio of the right tail (1 - percentile quantile) to the absolute value
    of the left tail (percentile quantile). Values > 1 indicate positive skew.

    Args:
        df: DataFrame with price data.
        window: Rolling window size.
        percentile: Tail percentile (e.g., 0.05 for 5th/95th).
        column: Price column.

    Returns:
        DataFrame with tail_ratio_{window} column appended.
    """

    def _compute(s: pl.Series) -> pl.Series:
        arr = s.to_numpy()
        n = len(arr)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            chunk = arr[i - window + 1 : i + 1]
            valid = chunk[~np.isnan(chunk)]
            if len(valid) < 10:
                continue
            right = np.quantile(valid, 1 - percentile)
            left = np.quantile(valid, percentile)
            if abs(left) < 1e-15:
                continue
            result[i] = abs(right / left)
        return pl.Series(result)

    return _apply_rolling_map(df, window, column, _compute, f"tail_ratio_{window}")


def rolling_jarque_bera(
    df: pl.DataFrame,
    window: int = 63,
    column: str = "close",
) -> pl.DataFrame:
    """Rolling Jarque-Bera test statistic.

    JB = n/6 * (S^2 + K^2/4) where S is skewness and K is excess kurtosis.
    High values indicate non-normal returns.

    Args:
        df: DataFrame with price data.
        window: Rolling window size.
        column: Price column.

    Returns:
        DataFrame with rolling_jb_{window} column appended.
    """
    from finasys.stats._utils import kurtosis as _kurtosis_np
    from finasys.stats._utils import skewness as _skewness_np

    def _compute(s: pl.Series) -> pl.Series:
        arr = s.to_numpy()
        n = len(arr)
        result = np.full(n, np.nan)
        for i in range(window - 1, n):
            chunk = arr[i - window + 1 : i + 1]
            valid = chunk[~np.isnan(chunk)]
            k = len(valid)
            if k < 4:
                continue
            s_val = _skewness_np(valid)
            k_val = _kurtosis_np(valid)
            result[i] = k / 6.0 * (s_val**2 + k_val**2 / 4.0)
        return pl.Series(result)

    return _apply_rolling_map(df, window, column, _compute, f"rolling_jb_{window}")


def zscore_returns(
    df: pl.DataFrame,
    window: int = 63,
    column: str = "close",
) -> pl.DataFrame:
    """Z-score of current return relative to rolling distribution.

    Measures how extreme the current return is relative to recent history.

    Args:
        df: DataFrame with price data.
        window: Rolling window size for mean/std computation.
        column: Price column.

    Returns:
        DataFrame with zscore_returns_{window} column appended.
    """
    col = pl.col(column)
    ret = col / col.shift(1) - 1
    rolling_mean = ret.rolling_mean(window_size=window)
    rolling_std = ret.rolling_std(window_size=window)

    expr = ((ret - rolling_mean) / rolling_std).alias(f"zscore_returns_{window}")
    return df.with_columns(symbol_aware(expr, df))
