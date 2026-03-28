"""Tests for stats shared utilities."""

import numpy as np
import polars as pl

from finasys.stats._utils import (
    apply_per_symbol,
    kurtosis,
    norm_ppf,
    price_to_returns_np,
    skewness,
)


class TestPriceToReturnsNp:
    def test_basic(self):
        prices = np.array([100.0, 102.0, 101.0, 105.0])
        rets = price_to_returns_np(prices)
        assert np.isnan(rets[0])
        assert abs(rets[1] - 0.02) < 1e-10

    def test_length_preserved(self):
        prices = np.array([10.0, 20.0, 30.0])
        assert len(price_to_returns_np(prices)) == 3


class TestSkewness:
    def test_symmetric(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(skewness(arr)) < 0.01

    def test_constant_returns_zero(self):
        arr = np.array([5.0, 5.0, 5.0, 5.0])
        assert skewness(arr) == 0.0


class TestKurtosis:
    def test_normal_like(self):
        rng = np.random.RandomState(42)
        arr = rng.normal(0, 1, 10000)
        k = kurtosis(arr)
        assert abs(k) < 0.5  # should be near 0 for normal

    def test_constant_returns_zero(self):
        arr = np.array([3.0, 3.0, 3.0, 3.0])
        assert kurtosis(arr) == 0.0


class TestNormPpf:
    def test_median(self):
        assert abs(norm_ppf(0.5)) < 1e-6

    def test_lower_tail(self):
        result = norm_ppf(0.05)
        assert result < -1.5

    def test_upper_tail(self):
        result = norm_ppf(0.95)
        assert result > 1.5

    def test_boundary_zero(self):
        assert norm_ppf(0.0) == 0.0

    def test_boundary_one(self):
        assert norm_ppf(1.0) == 0.0

    def test_symmetry(self):
        assert abs(norm_ppf(0.05) + norm_ppf(0.95)) < 1e-6


class TestApplyPerSymbol:
    def test_single_symbol(self, ohlcv_df):
        def _add_col(df):
            return df.with_columns(pl.lit(1).alias("test_col"))

        result = apply_per_symbol(ohlcv_df, _add_col)
        assert "test_col" in result.columns

    def test_multi_symbol(self, multi_symbol_df):
        def _add_col(df):
            return df.with_columns(pl.lit(1).alias("test_col"))

        result = apply_per_symbol(multi_symbol_df, _add_col)
        assert "test_col" in result.columns
        assert result.height == multi_symbol_df.height
