"""finasys.features -- Financial feature engineering.

Provides both a functional API and composable FeatureStep classes.

Functional API:
    import finasys as fs
    df = fs.features.rsi(df, period=14)
    df = fs.features.macd(df)
    df = fs.features.returns(df, periods=[1, 5, 21])

Composable pipeline:
    feature_set = fs.FeatureSet([
        fs.features.RSI(period=14),
        fs.features.MACD(),
        fs.features.Returns(periods=[1, 5, 21]),
    ])
    df = feature_set.transform(df)
"""

from __future__ import annotations

import polars as pl

from finasys.features.calendar import calendar_features
from finasys.features.cross import cross_percentile, cross_rank, cross_zscore
from finasys.features.distributions import (
    rolling_jarque_bera,
    rolling_kurtosis,
    rolling_skewness,
    tail_ratio,
    zscore_returns,
)

# --- Composable step classes (PascalCase) ---
from finasys.features.feature_set import (
    ATR,
    MACD,
    RSI,
    BollingerBands,
    BreakoutDetection,
    Calendar,
    ClassifyReturns,
    FeatureSet,
    FeatureStep,
    ForwardReturns,
    Lags,
    LogReturns,
    MarketState,
    Returns,
    RollingKurtosis,
    RollingSkewness,
    RollingStats,
    TailRatio,
    TrendStrength,
    TripleBarrier,
    VolAdjustedLabels,
    VolatilityRegime,
    ZscoreReturns,
)

# --- Functional API (lowercase) ---
from finasys.features.indicators import (
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
from finasys.features.lags import lags, validate_no_lookahead
from finasys.features.regime import (
    breakout_detection,
    market_state,
    trend_strength,
    volatility_regime,
)
from finasys.features.returns import (
    cumulative_returns,
    drawdown,
    log_returns,
    returns,
)
from finasys.features.rolling import rolling_stats
from finasys.features.targets import (
    classify_returns,
    forward_returns,
    triple_barrier_labels,
    volatility_adjusted_labels,
)


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
        from finasys.features.returns import returns as _returns

        df = _returns(df, periods=[1, 5, 21])

    if lags_ is not None:
        from finasys.features.lags import lags as _lags

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
    "volatility_regime",
    "trend_strength",
    "market_state",
    "breakout_detection",
    # Targets
    "forward_returns",
    "classify_returns",
    "triple_barrier_labels",
    "volatility_adjusted_labels",
    # Distributions
    "rolling_skewness",
    "rolling_kurtosis",
    "tail_ratio",
    "rolling_jarque_bera",
    "zscore_returns",
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
    "ForwardReturns",
    "ClassifyReturns",
    "TripleBarrier",
    "VolAdjustedLabels",
    "VolatilityRegime",
    "TrendStrength",
    "MarketState",
    "BreakoutDetection",
    "RollingSkewness",
    "RollingKurtosis",
    "TailRatio",
    "ZscoreReturns",
    "FeatureSet",
    "FeatureStep",
    # Convenience
    "add_all",
]
