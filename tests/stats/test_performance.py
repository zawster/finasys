"""Tests for performance metrics."""

import numpy as np
import polars as pl
import pytest


def _add_benchmark(df):
    """Helper to add a benchmark column to a DataFrame."""
    rng = np.random.RandomState(99)
    bench = [100.0]
    for _ in range(df.height - 1):
        bench.append(bench[-1] * (1 + rng.normal(0.0003, 0.012)))
    return df.with_columns(pl.Series("benchmark_close", bench))


class TestAlphaBeta:
    def test_scalar(self, ohlcv_df):
        from finasys.stats.performance import alpha_beta

        df = _add_benchmark(ohlcv_df)
        result = alpha_beta(df, benchmark_col="benchmark_close")
        assert isinstance(result, dict)
        assert "alpha" in result
        assert "beta" in result
        assert isinstance(result["alpha"], float)
        assert isinstance(result["beta"], float)

    def test_rolling(self, ohlcv_df):
        from finasys.stats.performance import alpha_beta

        df = _add_benchmark(ohlcv_df)
        result = alpha_beta(df, benchmark_col="benchmark_close", window=30)
        assert isinstance(result, pl.DataFrame)
        assert "alpha_30" in result.columns
        assert "beta_30" in result.columns

    def test_missing_benchmark(self, ohlcv_df):
        from finasys.stats.performance import alpha_beta

        with pytest.raises(ValueError, match="Benchmark column"):
            alpha_beta(ohlcv_df, benchmark_col="nonexistent")

    def test_short_df(self):
        from finasys.stats.performance import alpha_beta

        df = pl.DataFrame({"close": [100.0], "benchmark_close": [100.0]})
        result = alpha_beta(df, benchmark_col="benchmark_close")
        assert result == {"alpha": 0.0, "beta": 0.0}

    def test_rolling_multi_symbol(self, multi_symbol_df):
        from finasys.stats.performance import alpha_beta

        df = _add_benchmark(multi_symbol_df)
        result = alpha_beta(df, benchmark_col="benchmark_close", window=30)
        assert isinstance(result, pl.DataFrame)
        assert "alpha_30" in result.columns
        assert "beta_30" in result.columns


class TestInformationRatio:
    def test_basic(self, ohlcv_df):
        from finasys.stats.performance import information_ratio

        df = _add_benchmark(ohlcv_df)
        result = information_ratio(df, benchmark_col="benchmark_close")
        assert isinstance(result, float)

    def test_missing_benchmark(self, ohlcv_df):
        from finasys.stats.performance import information_ratio

        with pytest.raises(ValueError, match="Benchmark column"):
            information_ratio(ohlcv_df, benchmark_col="nonexistent")

    def test_short_df(self):
        from finasys.stats.performance import information_ratio

        df = pl.DataFrame({"close": [100.0], "benchmark_close": [100.0]})
        result = information_ratio(df, benchmark_col="benchmark_close")
        assert result == 0.0
