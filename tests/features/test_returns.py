"""Tests for return calculations."""

from alphakit.features import cumulative_returns, drawdown, log_returns, returns


class TestReturns:
    def test_single_period(self, simple_close_df):
        result = returns(simple_close_df, periods=1)
        assert "returns_1d" in result.columns
        # First return should be null (no previous data)
        assert result["returns_1d"][0] is None
        # Second: (102 - 100) / 100 = 0.02
        assert abs(result["returns_1d"][1] - 0.02) < 0.0001

    def test_multiple_periods(self, ohlcv_df):
        result = returns(ohlcv_df, periods=[1, 5, 21])
        assert "returns_1d" in result.columns
        assert "returns_5d" in result.columns
        assert "returns_21d" in result.columns


class TestLogReturns:
    def test_log_returns(self, simple_close_df):
        result = log_returns(simple_close_df, periods=1)
        assert "log_returns_1d" in result.columns


class TestCumulativeReturns:
    def test_cumulative_returns(self, simple_close_df):
        result = cumulative_returns(simple_close_df)
        assert "cumulative_returns" in result.columns
        # First value should be 0 (no change from start)
        assert abs(result["cumulative_returns"][0]) < 0.0001
        # Last: (110 - 100) / 100 = 0.10
        assert abs(result["cumulative_returns"][-1] - 0.10) < 0.0001


class TestDrawdown:
    def test_drawdown_columns(self, ohlcv_df):
        result = drawdown(ohlcv_df)
        assert "drawdown" in result.columns
        assert "max_drawdown" in result.columns

    def test_drawdown_always_negative(self, ohlcv_df):
        result = drawdown(ohlcv_df)
        dd = result["drawdown"].drop_nulls()
        assert (dd <= 0).all()
