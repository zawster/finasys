"""alphakit.features -- Financial feature engineering.

Provides both a functional API and composable FeatureStep classes.

Functional API:
    import alphakit as ak
    df = ak.features.rsi(df, period=14)
    df = ak.features.macd(df)
    df = ak.features.returns(df, periods=[1, 5, 21])

Composable pipeline:
    feature_set = ak.FeatureSet([
        ak.features.RSI(period=14),
        ak.features.MACD(),
        ak.features.Returns(periods=[1, 5, 21]),
    ])
    df = feature_set.transform(df)
"""

from __future__ import annotations

import polars as pl

from alphakit.features.calendar import calendar_features
from alphakit.features.cross import cross_percentile, cross_rank, cross_zscore

# --- Composable step classes (PascalCase) ---
from alphakit.features.feature_set import (
    ATR,
    MACD,
    RSI,
    BollingerBands,
    Calendar,
    FeatureSet,
    FeatureStep,
    Lags,
    LogReturns,
    Returns,
    RollingStats,
)

# --- Functional API (lowercase) ---
from alphakit.features.indicators import (
    adx,
    atr,
    bollinger,
    cci,
    ema,
    macd,
    mfi,
    momentum,
    obv,
    roc,
    rsi,
    sma,
    stochastic,
    vwap,
    williams_r,
)
from alphakit.features.lags import lags, validate_no_lookahead
from alphakit.features.returns import (
    cumulative_returns,
    drawdown,
    log_returns,
    returns,
)
from alphakit.features.rolling import rolling_stats


def add_all(
    df: pl.DataFrame,
    indicators: bool = True,
    returns_: bool = True,
    lags_: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    calendar: bool = False,
) -> pl.DataFrame:
    """Add a standard set of features in one call.

    A convenience function that applies commonly-used features.
    For fine-grained control, use individual functions or FeatureSet.

    Args:
        df: DataFrame with OHLCV data.
        indicators: Add RSI, MACD, Bollinger Bands, ATR, VWAP, OBV.
        returns_: Add simple returns for 1, 5, and 21 day periods.
        lags_: List of lag periods to create for the 'close' column.
        rolling_windows: Windows for rolling mean/std (default: [5, 21]).
        calendar: Add calendar features (day_of_week, month, etc.).
    """
    if indicators:
        # Check which columns are available for indicator calculation
        has_ohlcv = all(c in df.columns for c in ["open", "high", "low", "close", "volume"])

        df = rsi(df)
        df = macd(df)
        df = bollinger(df)
        df = sma(df, period=50)

        if has_ohlcv:
            df = atr(df)
            df = vwap(df)
            df = obv(df)
            df = stochastic(df)

    if returns_:
        from alphakit.features.returns import returns as _returns

        df = _returns(df, periods=[1, 5, 21])

    if lags_ is not None:
        from alphakit.features.lags import lags as _lags

        df = _lags(df, columns=["close"], lags=lags_)

    if rolling_windows is not None:
        windows = rolling_windows
    else:
        windows = None

    if windows is not None:
        df = rolling_stats(df, windows=windows)

    if calendar and "timestamp" in df.columns:
        df = calendar_features(df)

    return df


__all__ = [
    # Functional API
    "sma",
    "ema",
    "rsi",
    "macd",
    "bollinger",
    "atr",
    "vwap",
    "obv",
    "stochastic",
    "adx",
    "cci",
    "williams_r",
    "mfi",
    "roc",
    "momentum",
    "returns",
    "log_returns",
    "cumulative_returns",
    "drawdown",
    "rolling_stats",
    "lags",
    "validate_no_lookahead",
    "calendar_features",
    "cross_rank",
    "cross_percentile",
    "cross_zscore",
    # Composable steps
    "RSI",
    "MACD",
    "BollingerBands",
    "ATR",
    "Returns",
    "LogReturns",
    "RollingStats",
    "Lags",
    "Calendar",
    "FeatureSet",
    "FeatureStep",
    # Convenience
    "add_all",
]
