"""Tests for market regime features."""

import tempfile


def test_volatility_regime(ohlcv_df):
    from finasys.features import volatility_regime

    result = volatility_regime(ohlcv_df, fast_window=10, slow_window=30)

    assert "vol_ratio" in result.columns
    assert "vol_regime" in result.columns


def test_trend_strength(ohlcv_df):
    from finasys.features import trend_strength

    result = trend_strength(ohlcv_df, window=30)

    assert "hurst_30" in result.columns
    assert "trend_direction" in result.columns


def test_market_state(ohlcv_df):
    from finasys.features import market_state

    result = market_state(ohlcv_df, vol_window=30, trend_window=30)

    assert "market_state" in result.columns


def test_breakout_detection(ohlcv_df):
    from finasys.features import breakout_detection

    result = breakout_detection(ohlcv_df, window=20, n_std=2.0)

    assert "breakout_20" in result.columns
    assert "breakout_strength_20" in result.columns


def test_multi_symbol_regime(multi_symbol_df):
    from finasys.features import volatility_regime

    result = volatility_regime(multi_symbol_df, fast_window=10, slow_window=30)

    for sym in ["AAPL", "GOOGL"]:
        assert result.filter(result["symbol"] == sym)["vol_ratio"].len() > 0


def test_feature_set_regime_steps_roundtrip(ohlcv_df):
    from finasys.features import BreakoutDetection, FeatureSet, TrendStrength, VolatilityRegime

    pipeline = FeatureSet(
        [
            VolatilityRegime(fast_window=10, slow_window=30),
            TrendStrength(window=30),
            BreakoutDetection(window=20),
        ]
    )
    result1 = pipeline.transform(ohlcv_df)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        pipeline.save(f.name)
        loaded = FeatureSet.load(f.name)

    result2 = loaded.transform(ohlcv_df)
    assert result1.columns == result2.columns
