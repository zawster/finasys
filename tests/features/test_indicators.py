"""Tests for technical indicators."""

import polars as pl
import pytest

from alphakit.features import (
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


class TestSMA:
    def test_sma_output_column(self, ohlcv_df):
        result = sma(ohlcv_df, period=20)
        assert "sma_20" in result.columns

    def test_sma_first_values_null(self, ohlcv_df):
        result = sma(ohlcv_df, period=20)
        # First 19 values should be null
        assert result["sma_20"][:19].null_count() == 19

    def test_sma_value_correctness(self, simple_close_df):
        result = sma(simple_close_df, period=3)
        # SMA of [100, 102, 101] = 101.0
        val = result["sma_3"][2]
        assert abs(val - 101.0) < 0.01


class TestEMA:
    def test_ema_output_column(self, ohlcv_df):
        result = ema(ohlcv_df, period=20)
        assert "ema_20" in result.columns

    def test_ema_not_all_null(self, ohlcv_df):
        result = ema(ohlcv_df, period=20)
        assert result["ema_20"].null_count() < result.height


class TestRSI:
    def test_rsi_output_column(self, ohlcv_df):
        result = rsi(ohlcv_df, period=14)
        assert "rsi_14" in result.columns

    def test_rsi_range(self, ohlcv_df):
        result = rsi(ohlcv_df, period=14)
        non_null = result["rsi_14"].drop_nulls()
        assert non_null.min() >= 0
        assert non_null.max() <= 100


class TestMACD:
    def test_macd_output_columns(self, ohlcv_df):
        result = macd(ohlcv_df)
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns


class TestBollinger:
    def test_bollinger_output_columns(self, ohlcv_df):
        result = bollinger(ohlcv_df)
        assert "bb_middle" in result.columns
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns

    def test_bollinger_upper_above_lower(self, ohlcv_df):
        result = bollinger(ohlcv_df)
        valid = result.drop_nulls(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()


class TestATR:
    def test_atr_output_column(self, ohlcv_df):
        result = atr(ohlcv_df, period=14)
        assert "atr_14" in result.columns

    def test_atr_positive(self, ohlcv_df):
        result = atr(ohlcv_df)
        non_null = result["atr_14"].drop_nulls()
        assert (non_null > 0).all()


class TestVWAP:
    def test_vwap_output_column(self, ohlcv_df):
        result = vwap(ohlcv_df)
        assert "vwap" in result.columns


class TestOBV:
    def test_obv_output_column(self, ohlcv_df):
        result = obv(ohlcv_df)
        assert "obv" in result.columns


class TestStochastic:
    def test_stochastic_output_columns(self, ohlcv_df):
        result = stochastic(ohlcv_df)
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns

    def test_stochastic_range(self, ohlcv_df):
        result = stochastic(ohlcv_df)
        k = result["stoch_k"].drop_nulls()
        assert k.min() >= 0
        assert k.max() <= 100


class TestOtherIndicators:
    def test_cci(self, ohlcv_df):
        result = cci(ohlcv_df)
        assert "cci_20" in result.columns

    def test_williams_r(self, ohlcv_df):
        result = williams_r(ohlcv_df)
        assert "williams_r_14" in result.columns
        non_null = result["williams_r_14"].drop_nulls()
        assert non_null.min() >= -100
        assert non_null.max() <= 0

    def test_mfi(self, ohlcv_df):
        result = mfi(ohlcv_df)
        assert "mfi_14" in result.columns

    def test_roc(self, ohlcv_df):
        result = roc(ohlcv_df)
        assert "roc_10" in result.columns

    def test_momentum(self, ohlcv_df):
        result = momentum(ohlcv_df)
        assert "momentum_10" in result.columns

    def test_adx(self, ohlcv_df):
        from alphakit.features import adx
        result = adx(ohlcv_df)
        assert "adx_14" in result.columns
        assert "plus_di" in result.columns
        assert "minus_di" in result.columns
