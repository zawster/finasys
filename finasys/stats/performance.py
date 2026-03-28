"""Performance metrics for financial time series."""

from __future__ import annotations

import numpy as np
import polars as pl

from finasys.stats._utils import TRADING_DAYS, apply_per_symbol, price_to_returns_np


def alpha_beta(
    df: pl.DataFrame,
    benchmark_col: str = "benchmark_close",
    column: str = "close",
    window: int | None = None,
) -> pl.DataFrame | dict[str, float]:
    """CAPM alpha and beta vs a benchmark.

    Args:
        df: DataFrame with both asset and benchmark price columns.
        benchmark_col: Column name for benchmark prices.
        column: Column name for asset prices.
        window: If None, returns dict with alpha and beta.
                If set, appends rolling columns.

    Returns:
        Dict {"alpha": float, "beta": float} or DataFrame.
    """
    if benchmark_col not in df.columns:
        raise ValueError(f"Benchmark column '{benchmark_col}' not found in DataFrame")

    if window is None:
        asset_rets = (df[column] / df[column].shift(1) - 1).drop_nulls().to_numpy()
        bench_rets = (df[benchmark_col] / df[benchmark_col].shift(1) - 1).drop_nulls().to_numpy()

        min_len = min(len(asset_rets), len(bench_rets))
        if min_len < 2:
            return {"alpha": 0.0, "beta": 0.0}

        asset_rets = asset_rets[:min_len]
        bench_rets = bench_rets[:min_len]

        cov = np.cov(asset_rets, bench_rets)
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 1e-15 else 0.0
        alpha = float((asset_rets.mean() - beta * bench_rets.mean()) * TRADING_DAYS)

        return {"alpha": alpha, "beta": beta}

    return apply_per_symbol(df, _rolling_alpha_beta, window, benchmark_col, column)


def _rolling_alpha_beta(
    df: pl.DataFrame,
    window: int,
    benchmark_col: str,
    column: str,
) -> pl.DataFrame:
    """Compute rolling alpha/beta for a single-symbol DataFrame."""
    asset_rets = price_to_returns_np(df[column].to_numpy())
    bench_rets = price_to_returns_np(df[benchmark_col].to_numpy())
    n = len(asset_rets)

    alphas = np.full(n, np.nan)
    betas = np.full(n, np.nan)

    for i in range(window, n):
        a = asset_rets[i - window + 1 : i + 1]
        b = bench_rets[i - window + 1 : i + 1]
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if len(a) < 4:
            continue
        cov = np.cov(a, b)
        if cov[1, 1] < 1e-15:
            continue
        beta = cov[0, 1] / cov[1, 1]
        alpha = (a.mean() - beta * b.mean()) * TRADING_DAYS
        betas[i] = beta
        alphas[i] = alpha

    return df.with_columns(
        pl.Series(f"alpha_{window}", alphas),
        pl.Series(f"beta_{window}", betas),
    )


def information_ratio(
    df: pl.DataFrame,
    benchmark_col: str = "benchmark_close",
    column: str = "close",
) -> float:
    """Information ratio: active return / tracking error.

    Args:
        df: DataFrame with both asset and benchmark price columns.
        benchmark_col: Column name for benchmark prices.
        column: Column name for asset prices.

    Returns:
        Information ratio as a float.
    """
    if benchmark_col not in df.columns:
        raise ValueError(f"Benchmark column '{benchmark_col}' not found in DataFrame")

    asset_rets = (df[column] / df[column].shift(1) - 1).drop_nulls().to_numpy()
    bench_rets = (df[benchmark_col] / df[benchmark_col].shift(1) - 1).drop_nulls().to_numpy()

    min_len = min(len(asset_rets), len(bench_rets))
    if min_len < 2:
        return 0.0

    asset_rets = asset_rets[:min_len]
    bench_rets = bench_rets[:min_len]

    active_returns = asset_rets - bench_rets
    tracking_error = active_returns.std()

    if tracking_error < 1e-15:
        return 0.0

    return float(active_returns.mean() / tracking_error * np.sqrt(TRADING_DAYS))
