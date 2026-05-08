"""Tests for market regime features."""

import tempfile

import polars as pl
import pytest


def test_volatility_regime(ohlcv_df):
    from finasys.features import volatility_regime

    result = volatility_regime(ohlcv_df, fast_window=10, slow_window=30)

    assert "vol_ratio" in result.columns
    assert "vol_regime" in result.columns


def test_volatility_regime_validation(ohlcv_df):
    from finasys.features import volatility_regime

    with pytest.raises(ValueError, match="at least 2"):
        volatility_regime(ohlcv_df, fast_window=1, slow_window=30)

    with pytest.raises(ValueError, match="smaller"):
        volatility_regime(ohlcv_df, fast_window=30, slow_window=30)


def test_trend_strength(ohlcv_df):
    from finasys.features import trend_strength

    result = trend_strength(ohlcv_df, window=30)

    assert "hurst_30" in result.columns
    assert "trend_direction" in result.columns


def test_trend_strength_validation_and_downtrend(ohlcv_df):
    from finasys.features import trend_strength

    with pytest.raises(ValueError, match="at least 20"):
        trend_strength(ohlcv_df, window=10)

    down = ohlcv_df.with_columns((200.0 - pl.arange(0, pl.len()) * 2.0).alias("close"))
    result = trend_strength(down, window=30)
    assert result["trend_direction"].drop_nulls().item(-1) == -1


def test_trend_strength_multi_symbol(multi_symbol_df):
    from finasys.features import trend_strength

    result = trend_strength(multi_symbol_df, window=30)

    assert "hurst_30" in result.columns


def test_market_state(ohlcv_df):
    from finasys.features import market_state

    result = market_state(ohlcv_df, vol_window=30, trend_window=30)

    assert "market_state" in result.columns


def test_market_state_all_state_labels():
    from datetime import date, timedelta

    from finasys.features.regime import market_state

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(80)]
    close = [100.0 + i * 0.2 for i in range(80)]
    df = pl.DataFrame({"timestamp": dates, "close": close})
    result = market_state(df, vol_window=30, trend_window=30)

    assert result["market_state"].drop_nulls().len() > 0


def test_breakout_detection(ohlcv_df):
    from finasys.features import breakout_detection

    result = breakout_detection(ohlcv_df, window=20, n_std=2.0)

    assert "breakout_20" in result.columns
    assert "breakout_strength_20" in result.columns


def test_breakout_detection_validation(ohlcv_df):
    from finasys.features import breakout_detection

    with pytest.raises(ValueError, match="at least 2"):
        breakout_detection(ohlcv_df, window=1)


def test_multi_symbol_regime(multi_symbol_df):
    from finasys.features import volatility_regime

    result = volatility_regime(multi_symbol_df, fast_window=10, slow_window=30)

    for sym in ["AAPL", "GOOGL"]:
        assert result.filter(result["symbol"] == sym)["vol_ratio"].len() > 0


def test_feature_set_regime_steps_roundtrip(ohlcv_df):
    from finasys.features import BreakoutDetection, FeatureSet, MarketState, TrendStrength, VolatilityRegime

    pipeline = FeatureSet(
        [
            VolatilityRegime(fast_window=10, slow_window=30),
            TrendStrength(window=30),
            MarketState(vol_window=30, trend_window=30),
            BreakoutDetection(window=20),
        ]
    )
    result1 = pipeline.transform(ohlcv_df)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        pipeline.save(f.name)
        loaded = FeatureSet.load(f.name)

    result2 = loaded.transform(ohlcv_df)
    assert result1.columns == result2.columns
