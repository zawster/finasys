"""Tests for risk metrics."""

import polars as pl


class TestSharpeRatio:
    def test_scalar(self, ohlcv_df):
        from finasys.stats.risk import sharpe_ratio

        result = sharpe_ratio(ohlcv_df)
        assert isinstance(result, float)
        assert not result != result  # not NaN

    def test_rolling(self, ohlcv_df):
        from finasys.stats.risk import sharpe_ratio

        result = sharpe_ratio(ohlcv_df, window=21)
        assert isinstance(result, pl.DataFrame)
        assert "sharpe_21" in result.columns

    def test_with_risk_free(self, ohlcv_df):
        from finasys.stats.risk import sharpe_ratio

        result = sharpe_ratio(ohlcv_df, risk_free_rate=0.05)
        assert isinstance(result, float)


class TestSortinoRatio:
    def test_scalar(self, ohlcv_df):
        from finasys.stats.risk import sortino_ratio

        result = sortino_ratio(ohlcv_df)
        assert isinstance(result, float)

    def test_rolling(self, ohlcv_df):
        from finasys.stats.risk import sortino_ratio

        result = sortino_ratio(ohlcv_df, window=30)
        assert isinstance(result, pl.DataFrame)
        assert "sortino_30" in result.columns


class TestCalmarRatio:
    def test_basic(self, ohlcv_df):
        from finasys.stats.risk import calmar_ratio

        result = calmar_ratio(ohlcv_df)
        assert isinstance(result, float)


class TestValueAtRisk:
    def test_historical(self, ohlcv_df):
        from finasys.stats.risk import value_at_risk

        result = value_at_risk(ohlcv_df, confidence=0.95, method="historical")
        assert isinstance(result, float)
        assert result < 0  # VaR should be negative (representing loss)

    def test_cornish_fisher(self, ohlcv_df):
        from finasys.stats.risk import value_at_risk

        result = value_at_risk(ohlcv_df, confidence=0.95, method="cornish_fisher")
        assert isinstance(result, float)

    def test_rolling(self, ohlcv_df):
        from finasys.stats.risk import value_at_risk

        result = value_at_risk(ohlcv_df, window=30, method="historical")
        assert isinstance(result, pl.DataFrame)
        assert "var_30" in result.columns


class TestCVaR:
    def test_scalar(self, ohlcv_df):
        from finasys.stats.risk import cvar

        result = cvar(ohlcv_df, confidence=0.95)
        assert isinstance(result, float)
        assert result < 0  # CVaR should be negative

    def test_cvar_worse_than_var(self, ohlcv_df):
        from finasys.stats.risk import cvar, value_at_risk

        var = value_at_risk(ohlcv_df, confidence=0.95)
        cvar_val = cvar(ohlcv_df, confidence=0.95)
        assert cvar_val <= var  # CVaR is always worse (more negative)

    def test_rolling(self, ohlcv_df):
        from finasys.stats.risk import cvar

        result = cvar(ohlcv_df, window=30)
        assert isinstance(result, pl.DataFrame)
        assert "cvar_30" in result.columns


class TestMaxDrawdownDuration:
    def test_basic(self, ohlcv_df):
        from finasys.stats.risk import max_drawdown_duration

        result = max_drawdown_duration(ohlcv_df)
        assert "dd_duration" in result.columns
        assert "dd_max_duration" in result.columns
        # Durations should be non-negative
        assert (result["dd_duration"] >= 0).all()
        assert (result["dd_max_duration"] >= 0).all()
        # Max duration should be monotonically non-decreasing
        max_dur = result["dd_max_duration"].to_list()
        for i in range(1, len(max_dur)):
            assert max_dur[i] >= max_dur[i - 1]

    def test_multi_symbol(self, multi_symbol_df):
        from finasys.stats.risk import max_drawdown_duration

        result = max_drawdown_duration(multi_symbol_df)
        assert "dd_duration" in result.columns
