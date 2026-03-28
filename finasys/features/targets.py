"""Target/label engineering for supervised financial ML.

Provides forward-looking return calculations and classification labels
with explicit look-ahead awareness. These columns are intended as ML targets,
not features -- they must be dropped before inference.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from finasys.features.utils import has_multi_symbols, symbol_aware


def forward_returns(
    df: pl.DataFrame,
    periods: list[int] | int = 1,
    column: str = "close",
) -> pl.DataFrame:
    """Forward-looking returns over given period(s).

    These are TARGET columns for supervised ML. They use future data
    and must be dropped before inference/production scoring.

    Symbol-aware for multi-symbol DataFrames.

    Args:
        df: DataFrame with price data.
        periods: Single period or list of periods (e.g., [1, 5, 21]).
        column: Price column to compute forward returns from.

    Returns:
        DataFrame with forward return columns appended (e.g., fwd_return_1d).
    """
    if isinstance(periods, int):
        periods = [periods]

    col = pl.col(column)
    exprs = []
    for p in periods:
        expr = (col.shift(-p) / col - 1).alias(f"fwd_return_{p}d")
        exprs.append(symbol_aware(expr, df))

    return df.with_columns(exprs)


def classify_returns(
    df: pl.DataFrame,
    period: int = 5,
    thresholds: tuple[float, float] = (-0.01, 0.01),
    column: str = "close",
) -> pl.DataFrame:
    """Classify forward returns into ternary labels.

    Labels:
        -1: down (forward return < lower threshold)
         0: flat (forward return between thresholds)
         1: up   (forward return > upper threshold)

    Args:
        df: DataFrame with price data.
        period: Forward-looking period in bars.
        thresholds: (lower, upper) thresholds for classification.
        column: Price column.

    Returns:
        DataFrame with label column appended (e.g., label_5d).
    """
    col = pl.col(column)
    fwd_ret = col.shift(-period) / col - 1

    label_expr = (
        pl.when(fwd_ret < thresholds[0])
        .then(pl.lit(-1))
        .when(fwd_ret > thresholds[1])
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias(f"label_{period}d")
    )

    return df.with_columns(symbol_aware(label_expr, df))


def triple_barrier_labels(
    df: pl.DataFrame,
    profit_take: float = 0.02,
    stop_loss: float = 0.02,
    max_holding: int = 10,
    column: str = "close",
) -> pl.DataFrame:
    """Lopez de Prado triple-barrier labeling method.

    Three barriers:
        - Upper barrier: price rises by profit_take fraction (label = 1)
        - Lower barrier: price falls by stop_loss fraction (label = -1)
        - Vertical barrier: max_holding bars elapse (label = sign of return)

    Appends columns:
        - tb_label: 1 (profit take), -1 (stop loss), 0 (neutral at expiry)
        - tb_duration: number of bars held
        - tb_return: actual return at exit

    Args:
        df: DataFrame with price data.
        profit_take: Fraction for upper barrier (e.g., 0.02 = 2%).
        stop_loss: Fraction for lower barrier (e.g., 0.02 = 2%).
        max_holding: Maximum holding period in bars.
        column: Price column.

    Returns:
        DataFrame with triple-barrier columns appended.
    """
    if has_multi_symbols(df):
        # Process each symbol separately
        frames = []
        for sym in df["symbol"].unique().sort().to_list():
            sym_df = df.filter(pl.col("symbol") == sym)
            sym_df = _apply_triple_barrier(sym_df, profit_take, stop_loss, max_holding, column)
            frames.append(sym_df)
        return pl.concat(frames).sort(["timestamp", "symbol"])
    else:
        return _apply_triple_barrier(df, profit_take, stop_loss, max_holding, column)


def _apply_triple_barrier(
    df: pl.DataFrame,
    profit_take: float,
    stop_loss: float,
    max_holding: int,
    column: str,
) -> pl.DataFrame:
    """Apply triple-barrier labeling to a single-symbol DataFrame."""
    prices = df[column].to_numpy()
    n = len(prices)

    labels = np.full(n, np.nan)
    durations = np.full(n, np.nan)
    exit_returns = np.full(n, np.nan)

    for i in range(n):
        entry_price = prices[i]
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        upper = entry_price * (1 + profit_take)
        lower = entry_price * (1 - stop_loss)
        end = min(i + max_holding, n - 1)

        hit_label = 0
        hit_bar = end
        hit_return = 0.0

        for j in range(i + 1, end + 1):
            if np.isnan(prices[j]):
                continue
            if prices[j] >= upper:
                hit_label = 1
                hit_bar = j
                hit_return = prices[j] / entry_price - 1
                break
            elif prices[j] <= lower:
                hit_label = -1
                hit_bar = j
                hit_return = prices[j] / entry_price - 1
                break
        else:
            # Vertical barrier hit
            if end < n and not np.isnan(prices[end]):
                hit_return = prices[end] / entry_price - 1
                hit_label = int(np.sign(hit_return))
                hit_bar = end

        labels[i] = hit_label
        durations[i] = hit_bar - i
        exit_returns[i] = hit_return

    # NaN entries (from skipped zero/NaN prices) become null after cast
    tb_label = pl.Series("tb_label", labels)
    tb_duration = pl.Series("tb_duration", durations)
    mask = tb_label.is_nan()
    tb_label = tb_label.set(mask, None).cast(pl.Int32)
    tb_duration = tb_duration.set(mask, None).cast(pl.Int32)

    return df.with_columns(
        tb_label,
        tb_duration,
        pl.Series("tb_return", exit_returns),
    )


def volatility_adjusted_labels(
    df: pl.DataFrame,
    period: int = 5,
    vol_window: int = 21,
    vol_multiplier: float = 1.0,
    column: str = "close",
) -> pl.DataFrame:
    """Classify forward returns relative to rolling volatility.

    Thresholds adapt to the current volatility regime:
        up:   fwd_return > vol_multiplier * rolling_std
        down: fwd_return < -vol_multiplier * rolling_std
        flat: otherwise

    This is more robust than fixed thresholds across different
    volatility regimes.

    Args:
        df: DataFrame with price data.
        period: Forward-looking period in bars.
        vol_window: Window for rolling volatility computation.
        vol_multiplier: Multiplier for volatility threshold.
        column: Price column.

    Returns:
        DataFrame with vol_label column appended (e.g., vol_label_5d).
    """
    col = pl.col(column)

    # Daily returns for volatility
    daily_ret = col / col.shift(1) - 1
    rolling_vol = daily_ret.rolling_std(window_size=vol_window)

    # Forward returns
    fwd_ret = col.shift(-period) / col - 1

    # Dynamic thresholds
    upper_thresh = vol_multiplier * rolling_vol
    lower_thresh = -vol_multiplier * rolling_vol

    label_expr = (
        pl.when(fwd_ret > upper_thresh)
        .then(pl.lit(1))
        .when(fwd_ret < lower_thresh)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias(f"vol_label_{period}d")
    )

    return df.with_columns(symbol_aware(label_expr, df))
