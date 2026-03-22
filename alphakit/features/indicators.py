"""Technical indicators implemented in pure Polars expressions.

Each function takes a Polars DataFrame and returns a new DataFrame with
the indicator column(s) appended. No mutation -- always returns a copy.

All implementations use Polars' expression API for maximum performance.
No dependency on ta-lib or pandas-ta.

All functions are symbol-aware: in multi-symbol DataFrames, indicators
are computed within each symbol using .over("symbol").
"""

from __future__ import annotations

import polars as pl

from alphakit.features.utils import symbol_aware


def sma(df: pl.DataFrame, period: int = 20, column: str = "close") -> pl.DataFrame:
    """Simple Moving Average.

    The SMA is the unweighted mean of the previous `period` data points.
    """
    expr = pl.col(column).rolling_mean(window_size=period).alias(f"sma_{period}")
    return df.with_columns(symbol_aware(expr, df))


def ema(df: pl.DataFrame, period: int = 20, column: str = "close") -> pl.DataFrame:
    """Exponential Moving Average.

    The EMA gives more weight to recent prices. Uses the standard
    smoothing factor: alpha = 2 / (period + 1).
    """
    expr = pl.col(column).ewm_mean(span=period, adjust=False).alias(f"ema_{period}")
    return df.with_columns(symbol_aware(expr, df))


def rsi(df: pl.DataFrame, period: int = 14, column: str = "close") -> pl.DataFrame:
    """Relative Strength Index (RSI).

    RSI measures the speed and magnitude of price changes.
    Values above 70 suggest overbought, below 30 suggest oversold.

    Implementation:
    1. Calculate price changes (deltas)
    2. Separate gains (positive deltas) and losses (negative deltas)
    3. Compute EWM mean of gains and losses
    4. RSI = 100 - (100 / (1 + avg_gain / avg_loss))
    """
    col = pl.col(column)
    delta = col.diff()

    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)

    expr = (
        pl.lit(100)
        - pl.lit(100)
        / (pl.lit(1) + gain.ewm_mean(span=period, adjust=False) / loss.ewm_mean(span=period, adjust=False))
    ).alias(f"rsi_{period}")

    return df.with_columns(symbol_aware(expr, df))


def macd(
    df: pl.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pl.DataFrame:
    """Moving Average Convergence Divergence (MACD).

    Appends three columns:
    - macd_line: difference between fast and slow EMA
    - macd_signal: EMA of the MACD line
    - macd_hist: MACD line minus signal line (histogram)
    """
    col = pl.col(column)
    fast_ema = col.ewm_mean(span=fast, adjust=False)
    slow_ema = col.ewm_mean(span=slow, adjust=False)
    macd_line = fast_ema - slow_ema

    line_expr = symbol_aware(macd_line.alias("macd_line"), df)
    signal_expr = symbol_aware(macd_line.ewm_mean(span=signal, adjust=False).alias("macd_signal"), df)
    hist_expr = symbol_aware((macd_line - macd_line.ewm_mean(span=signal, adjust=False)).alias("macd_hist"), df)

    return df.with_columns(line_expr, signal_expr, hist_expr)


def bollinger(
    df: pl.DataFrame,
    period: int = 20,
    std: float = 2.0,
    column: str = "close",
) -> pl.DataFrame:
    """Bollinger Bands.

    Appends three columns:
    - bb_middle: SMA of the price
    - bb_upper: middle + std * rolling standard deviation
    - bb_lower: middle - std * rolling standard deviation
    """
    col = pl.col(column)
    middle = col.rolling_mean(window_size=period)
    rolling_std = col.rolling_std(window_size=period)

    return df.with_columns(
        symbol_aware(middle.alias("bb_middle"), df),
        symbol_aware((middle + std * rolling_std).alias("bb_upper"), df),
        symbol_aware((middle - std * rolling_std).alias("bb_lower"), df),
    )


def atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Average True Range (ATR).

    Measures market volatility using the greatest of:
    - Current high - current low
    - |Current high - previous close|
    - |Current low - previous close|
    """
    high = pl.col("high")
    low = pl.col("low")
    prev_close = pl.col("close").shift(1)

    tr = pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    )

    expr = tr.ewm_mean(span=period, adjust=False).alias(f"atr_{period}")
    return df.with_columns(symbol_aware(expr, df))


def vwap(df: pl.DataFrame) -> pl.DataFrame:
    """Volume Weighted Average Price (VWAP).

    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    where typical_price = (high + low + close) / 3
    """
    typical = (pl.col("high") + pl.col("low") + pl.col("close")) / 3
    tp_vol = typical * pl.col("volume")

    expr = (tp_vol.cum_sum() / pl.col("volume").cum_sum()).alias("vwap")
    return df.with_columns(symbol_aware(expr, df))


def obv(df: pl.DataFrame) -> pl.DataFrame:
    """On-Balance Volume (OBV).

    Cumulative sum of volume, adding volume on up days
    and subtracting on down days.
    """
    direction = (
        pl.when(pl.col("close") > pl.col("close").shift(1))
        .then(1)
        .when(pl.col("close") < pl.col("close").shift(1))
        .then(-1)
        .otherwise(0)
    )

    expr = (direction * pl.col("volume")).cum_sum().alias("obv")
    return df.with_columns(symbol_aware(expr, df))


def stochastic(df: pl.DataFrame, k_period: int = 14, d_period: int = 3) -> pl.DataFrame:
    """Stochastic Oscillator (%K and %D).

    %K = (close - lowest_low) / (highest_high - lowest_low) * 100
    %D = SMA of %K
    """
    lowest = pl.col("low").rolling_min(window_size=k_period)
    highest = pl.col("high").rolling_max(window_size=k_period)

    stoch_k = (pl.col("close") - lowest) / (highest - lowest) * 100

    return df.with_columns(
        symbol_aware(stoch_k.alias("stoch_k"), df),
        symbol_aware(stoch_k.rolling_mean(window_size=d_period).alias("stoch_d"), df),
    )


def adx(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Average Directional Index (ADX).

    Measures trend strength regardless of direction.
    Values above 25 indicate a strong trend.
    """
    high = pl.col("high")
    low = pl.col("low")
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = pl.when((high - prev_high) > (prev_low - low)).then((high - prev_high).clip(lower_bound=0)).otherwise(0)
    minus_dm = pl.when((prev_low - low) > (high - prev_high)).then((prev_low - low).clip(lower_bound=0)).otherwise(0)

    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    )

    atr_smooth = tr.ewm_mean(span=period, adjust=False)
    plus_di = plus_dm.ewm_mean(span=period, adjust=False) / atr_smooth * 100
    minus_di = minus_dm.ewm_mean(span=period, adjust=False) / atr_smooth * 100

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100

    return df.with_columns(
        symbol_aware(plus_di.alias("plus_di"), df),
        symbol_aware(minus_di.alias("minus_di"), df),
        symbol_aware(dx.ewm_mean(span=period, adjust=False).alias(f"adx_{period}"), df),
    )


def cci(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """Commodity Channel Index (CCI).

    CCI = (typical_price - SMA(typical_price)) / (0.015 * mean_deviation)
    """
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3
    tp_sma = tp.rolling_mean(window_size=period)
    tp_std = tp.rolling_std(window_size=period)

    expr = ((tp - tp_sma) / (0.015 * tp_std)).alias(f"cci_{period}")
    return df.with_columns(symbol_aware(expr, df))


def williams_r(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Williams %R.

    Similar to stochastic but inverted. Ranges from -100 to 0.
    """
    highest = pl.col("high").rolling_max(window_size=period)
    lowest = pl.col("low").rolling_min(window_size=period)

    expr = ((highest - pl.col("close")) / (highest - lowest) * -100).alias(f"williams_r_{period}")
    return df.with_columns(symbol_aware(expr, df))


def mfi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Money Flow Index (MFI).

    Volume-weighted RSI. Combines price and volume to measure
    buying/selling pressure.
    """
    tp = (pl.col("high") + pl.col("low") + pl.col("close")) / 3
    raw_mf = tp * pl.col("volume")

    tp_diff = tp.diff()
    pos_mf = pl.when(tp_diff > 0).then(raw_mf).otherwise(0)
    neg_mf = pl.when(tp_diff < 0).then(raw_mf).otherwise(0)

    pos_sum = pos_mf.rolling_sum(window_size=period)
    neg_sum = neg_mf.rolling_sum(window_size=period)

    expr = (100 - 100 / (1 + pos_sum / neg_sum)).alias(f"mfi_{period}")
    return df.with_columns(symbol_aware(expr, df))


def roc(df: pl.DataFrame, period: int = 10, column: str = "close") -> pl.DataFrame:
    """Rate of Change (ROC).

    Percentage change from `period` bars ago.
    """
    col = pl.col(column)
    expr = ((col - col.shift(period)) / col.shift(period) * 100).alias(f"roc_{period}")
    return df.with_columns(symbol_aware(expr, df))


def momentum(df: pl.DataFrame, period: int = 10, column: str = "close") -> pl.DataFrame:
    """Momentum.

    Simple difference between current price and price `period` bars ago.
    """
    col = pl.col(column)
    expr = (col - col.shift(period)).alias(f"momentum_{period}")
    return df.with_columns(symbol_aware(expr, df))
