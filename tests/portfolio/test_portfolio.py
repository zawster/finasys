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
    raw_result = covariance_matrix(multi_symbol_df, annualize=False)

    assert result.height == 2
    assert "AAPL" in result.columns
    assert raw_result.height == 2


def test_correlation_matrix_spearman(multi_symbol_df):
    from finasys.portfolio import correlation_matrix

    result = correlation_matrix(multi_symbol_df, method="spearman")

    assert result.height == 2


def test_correlation_matrix_invalid_method(multi_symbol_df):
    from finasys.portfolio import correlation_matrix

    with pytest.raises(ValueError, match="method"):
        correlation_matrix(multi_symbol_df, method="kendall")


def test_rolling_correlation(multi_symbol_df):
    from finasys.portfolio import rolling_correlation

    result = rolling_correlation(multi_symbol_df, "AAPL", "GOOGL", window=10)

    assert "rolling_corr_AAPL_GOOGL_10" in result.columns
    assert result.height == multi_symbol_df["timestamp"].n_unique()


def test_rolling_correlation_validation(multi_symbol_df):
    from finasys.portfolio import rolling_correlation

    with pytest.raises(ValueError, match="at least 2"):
        rolling_correlation(multi_symbol_df, "AAPL", "GOOGL", window=1)

    with pytest.raises(ValueError, match="Both symbols"):
        rolling_correlation(multi_symbol_df, "AAPL", "MSFT", window=10)


def test_portfolio_returns(multi_symbol_df):
    from finasys.portfolio import portfolio_returns

    result = portfolio_returns(multi_symbol_df, {"AAPL": 0.6, "GOOGL": 0.4})

    assert result.columns == ["timestamp", "portfolio_returns"]
    assert result.height == multi_symbol_df["timestamp"].n_unique()


def test_portfolio_returns_validation(multi_symbol_df):
    from finasys.portfolio import portfolio_returns

    with pytest.raises(ValueError, match="must not be empty"):
        portfolio_returns(multi_symbol_df, {})

    with pytest.raises(ValueError, match="not found"):
        portfolio_returns(multi_symbol_df, {"MSFT": 1.0})

    with pytest.raises(ValueError, match="sum to zero"):
        portfolio_returns(multi_symbol_df, {"AAPL": 1.0, "GOOGL": -1.0})


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


def test_missing_required_column(multi_symbol_df):
    from finasys.portfolio import correlation_matrix

    with pytest.raises(ValueError, match="Missing required columns"):
        correlation_matrix(multi_symbol_df.drop("close"))


def test_matrix_outputs_with_insufficient_complete_returns():
    from datetime import date

    from finasys.portfolio import correlation_matrix, covariance_matrix, minimum_variance_weights

    df = pl.DataFrame(
        {
            "timestamp": [date(2024, 1, 1), date(2024, 1, 1)],
            "symbol": ["AAPL", "GOOGL"],
            "close": [100.0, 200.0],
        }
    )

    assert correlation_matrix(df).height == 2
    assert covariance_matrix(df).height == 2
    with pytest.raises(ValueError, match="At least two"):
        minimum_variance_weights(df)
