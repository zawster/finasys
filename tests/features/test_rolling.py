"""Tests for rolling statistics."""

import pytest

from finasys.features import rolling_stats


def test_rolling_stats_default(ohlcv_df):
    result = rolling_stats(ohlcv_df, windows=21)
    assert "rolling_mean_21" in result.columns
    assert "rolling_std_21" in result.columns


def test_rolling_stats_multiple_windows(ohlcv_df):
    result = rolling_stats(ohlcv_df, windows=[5, 21])
    assert "rolling_mean_5" in result.columns
    assert "rolling_mean_21" in result.columns
    assert "rolling_std_5" in result.columns
    assert "rolling_std_21" in result.columns


def test_rolling_stats_custom_stats(ohlcv_df):
    result = rolling_stats(ohlcv_df, windows=10, stats=["min", "max", "zscore"])
    assert "rolling_min_10" in result.columns
    assert "rolling_max_10" in result.columns
    assert "rolling_zscore_10" in result.columns


def test_rolling_stats_unknown_stat(ohlcv_df):
    with pytest.raises(ValueError, match="Unknown stat"):
        rolling_stats(ohlcv_df, windows=10, stats=["unknown"])
