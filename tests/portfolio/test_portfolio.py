"""Tests for portfolio analytics."""

import polars as pl
import pytest


def test_correlation_matrix(multi_symbol_df):
    from finasys.portfolio import correlation_matrix

    result = correlation_matrix(multi_symbol_df)

    assert result.height == 2
    assert "symbol" in result.columns
    assert {"AAPL", "GOOGL"}.issubset(set(result.columns))
    assert result.filter(pl.col("symbol") == "AAPL")["AAPL"][0] == pytest.approx(1.0)


def test_covariance_matrix(multi_symbol_df):
    from finasys.portfolio import covariance_matrix

    result = covariance_matrix(multi_symbol_df)

    assert result.height == 2
    assert "AAPL" in result.columns


def test_rolling_correlation(multi_symbol_df):
    from finasys.portfolio import rolling_correlation

    result = rolling_correlation(multi_symbol_df, "AAPL", "GOOGL", window=10)

    assert "rolling_corr_AAPL_GOOGL_10" in result.columns
    assert result.height == multi_symbol_df["timestamp"].n_unique()


def test_portfolio_returns(multi_symbol_df):
    from finasys.portfolio import portfolio_returns

    result = portfolio_returns(multi_symbol_df, {"AAPL": 0.6, "GOOGL": 0.4})

    assert result.columns == ["timestamp", "portfolio_returns"]
    assert result.height == multi_symbol_df["timestamp"].n_unique()


def test_equal_weight_returns(multi_symbol_df):
    from finasys.portfolio import equal_weight_returns

    result = equal_weight_returns(multi_symbol_df)

    assert "portfolio_returns" in result.columns


def test_minimum_variance_weights(multi_symbol_df):
    from finasys.portfolio import minimum_variance_weights

    weights = minimum_variance_weights(multi_symbol_df)

    assert set(weights) == {"AAPL", "GOOGL"}
    assert sum(weights.values()) == pytest.approx(1.0)


def test_requires_multi_symbol(ohlcv_df):
    from finasys.portfolio import correlation_matrix

    with pytest.raises(ValueError, match="at least two symbols"):
        correlation_matrix(ohlcv_df)
