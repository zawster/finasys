"""Market regime detection features."""

from __future__ import annotations

import numpy as np
import polars as pl

from finasys.features.utils import has_multi_symbols, symbol_aware

__all__ = [
    "volatility_regime",
    "trend_strength",
    "market_state",
    "breakout_detection",
]


def volatility_regime(
    df: pl.DataFrame,
    fast_window: int = 21,
    slow_window: int = 63,
    column: str = "close",
) -> pl.DataFrame:
    """Classify high/low volatility regime using fast/slow return volatility."""
    if fast_window < 2 or slow_window < 2:
        raise ValueError("fast_window and slow_window must be at least 2")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")

    ret = pl.col(column) / pl.col(column).shift(1) - 1
    fast_vol = ret.rolling_std(window_size=fast_window)
    slow_vol = ret.rolling_std(window_size=slow_window)
    ratio = fast_vol / slow_vol

    return df.with_columns(
        symbol_aware(ratio.alias("vol_ratio"), df),
        symbol_aware((ratio > 1.0).cast(pl.Int8).alias("vol_regime"), df),
    )


def _hurst_window(values: np.ndarray) -> float:
    values = values[~np.isnan(values)]
    if len(values) < 20:
        return np.nan
    lags = np.array([2, 4, 8, 16])
    valid_lags = lags[lags < len(values) // 2]
    if len(valid_lags) < 2:
        return np.nan

    tau = []
    for lag in valid_lags:
        diff = values[lag:] - values[:-lag]
        std = np.std(diff)
        if std <= 1e-15:
            return 0.5
        tau.append(std)
    slope = np.polyfit(np.log(valid_lags), np.log(tau), 1)[0]
    return float(max(0.0, min(1.0, slope)))


def _apply_hurst(df: pl.DataFrame, window: int, column: str) -> pl.DataFrame:
    values = df[column].to_numpy()
    result = np.full(len(values), np.nan)
    for i in range(window - 1, len(values)):
        result[i] = _hurst_window(values[i - window + 1 : i + 1])
    return df.with_columns(pl.Series(f"hurst_{window}", result))


def trend_strength(
    df: pl.DataFrame,
    window: int = 63,
    column: str = "close",
) -> pl.DataFrame:
    """Append Hurst approximation and directional trend classification."""
    if window < 20:
        raise ValueError("window must be at least 20")

    if has_multi_symbols(df):
        frames = []
        for sym in df["symbol"].unique().sort().to_list():
            frames.append(_apply_hurst(df.filter(pl.col("symbol") == sym), window, column))
        result = pl.concat(frames).sort(["timestamp", "symbol"])
    else:
        result = _apply_hurst(df, window, column)

    change = pl.col(column) / pl.col(column).shift(window) - 1
    direction = (
        pl.when(change > 0.02)
        .then(pl.lit(1))
        .when(change < -0.02)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("trend_direction")
    )
    return result.with_columns(symbol_aware(direction, result))


def market_state(
    df: pl.DataFrame,
    vol_window: int = 21,
    trend_window: int = 63,
    column: str = "close",
) -> pl.DataFrame:
    """Combine volatility and trend classifications into market_state."""
    result = volatility_regime(df, fast_window=max(2, vol_window // 2), slow_window=vol_window, column=column)
    result = trend_strength(result, window=trend_window, column=column)

    state = (
        pl.when((pl.col("trend_direction") != 0) & (pl.col("vol_regime") == 1))
        .then(pl.lit("trending_high_vol"))
        .when((pl.col("trend_direction") != 0) & (pl.col("vol_regime") == 0))
        .then(pl.lit("trending_low_vol"))
        .when((pl.col("trend_direction") == 0) & (pl.col("vol_regime") == 1))
        .then(pl.lit("ranging_high_vol"))
        .otherwise(pl.lit("ranging_low_vol"))
        .alias("market_state")
    )
    return result.with_columns(state)


def breakout_detection(
    df: pl.DataFrame,
    window: int = 20,
    n_std: float = 2.0,
    column: str = "close",
) -> pl.DataFrame:
    """Flag price breakouts relative to a rolling mean and standard deviation."""
    if window < 2:
        raise ValueError("window must be at least 2")

    col = pl.col(column)
    mean = col.rolling_mean(window_size=window)
    std = col.rolling_std(window_size=window)
    upper = mean + n_std * std
    lower = mean - n_std * std
    strength = ((col - mean) / std).alias(f"breakout_strength_{window}")
    flag = ((col > upper) | (col < lower)).fill_null(False).cast(pl.Int8).alias(f"breakout_{window}")

    return df.with_columns(
        symbol_aware(strength, df),
        symbol_aware(flag, df),
    )
