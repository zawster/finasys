"""Tests for the add_all convenience function."""

from alphakit.features import add_all


def test_add_all_default(ohlcv_df):
    result = add_all(ohlcv_df)

    # Should have RSI, MACD, Bollinger, SMA, ATR, VWAP, OBV, stochastic
    assert "rsi_14" in result.columns
    assert "macd_line" in result.columns
    assert "bb_upper" in result.columns
    assert "sma_50" in result.columns
    assert "atr_14" in result.columns
    assert "vwap" in result.columns

    # Returns
    assert "returns_1d" in result.columns
    assert "returns_5d" in result.columns
    assert "returns_21d" in result.columns


def test_add_all_with_lags(ohlcv_df):
    result = add_all(ohlcv_df, lags_=[1, 5])
    assert "close_lag_1" in result.columns
    assert "close_lag_5" in result.columns


def test_add_all_no_indicators(ohlcv_df):
    result = add_all(ohlcv_df, indicators=False, returns_=True)
    assert "rsi_14" not in result.columns
    assert "returns_1d" in result.columns
