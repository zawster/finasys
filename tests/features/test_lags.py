"""Tests for lag features and look-ahead bias protection."""

import polars as pl
import pytest

from finasys.features import lags, validate_no_lookahead


def test_lags_single(simple_close_df):
    result = lags(simple_close_df, columns="close", lags=1)
    assert "close_lag_1" in result.columns
    # First value should be null
    assert result["close_lag_1"][0] is None
    # Second value should be the first close (100.0)
    assert result["close_lag_1"][1] == 100.0


def test_lags_multiple(ohlcv_df):
    result = lags(ohlcv_df, columns=["close", "volume"], lags=[1, 3, 5])
    assert "close_lag_1" in result.columns
    assert "close_lag_5" in result.columns
    assert "volume_lag_3" in result.columns


def test_lags_rejects_negative():
    """Negative lags would cause look-ahead bias."""
    df = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="positive"):
        lags(df, columns="close", lags=-1)


def test_lags_rejects_zero():
    df = pl.DataFrame({"close": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="positive"):
        lags(df, columns="close", lags=0)


def test_validate_no_lookahead_passes(ohlcv_df):
    """Lag features should pass look-ahead validation."""
    from finasys.features import rsi

    full = rsi(ohlcv_df, period=14)
    partial = rsi(ohlcv_df.head(50), period=14)

    assert validate_no_lookahead(full, partial, ["rsi_14"]) is True
