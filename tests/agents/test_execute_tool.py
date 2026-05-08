"""Tests for agents.execute_tool() and context edge cases."""


class TestExecuteTool:
    """Test execute_tool with mocked data (no network)."""

    def test_unknown_tool(self):
        from finasys.agents.tools import execute_tool

        result = execute_tool("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_lookup_price(self, ohlcv_df, tmp_path, monkeypatch):
        """Mock fs.load to avoid network calls."""
        import finasys as fs

        monkeypatch.setattr(fs, "load", lambda *a, **kw: ohlcv_df)
        from finasys.agents.tools import execute_tool

        result = execute_tool("lookup_price", {"symbol": "TEST"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_technical_indicators(self, ohlcv_df, monkeypatch):
        import finasys as fs

        monkeypatch.setattr(fs, "load", lambda *a, **kw: ohlcv_df)
        from finasys.agents.tools import execute_tool

        result = execute_tool(
            "get_technical_indicators",
            {"symbol": "TEST", "indicators": ["rsi", "macd"]},
        )
        assert isinstance(result, str)

    def test_get_technical_indicators_ignores_unknown_indicator(self, ohlcv_df, monkeypatch):
        import finasys as fs

        monkeypatch.setattr(fs, "load", lambda *a, **kw: ohlcv_df)
        from finasys.agents.tools import execute_tool

        result = execute_tool(
            "get_technical_indicators",
            {"symbol": "TEST", "indicators": ["rsi", "not_a_real_indicator"]},
        )
        assert isinstance(result, str)

    def test_compare_symbols(self, multi_symbol_df, monkeypatch):
        import finasys as fs

        monkeypatch.setattr(fs, "load", lambda *a, **kw: multi_symbol_df)
        from finasys.agents.tools import execute_tool

        result = execute_tool("compare_symbols", {"symbols": ["AAPL", "GOOGL"]})
        assert isinstance(result, str)

    def test_get_summary(self, ohlcv_df, monkeypatch):
        import finasys as fs

        monkeypatch.setattr(fs, "load", lambda *a, **kw: ohlcv_df)
        from finasys.agents.tools import execute_tool

        result = execute_tool("get_summary", {"symbol": "TEST", "days": 50})
        assert isinstance(result, str)
        assert "$" in result

    def test_assess_risk(self, ohlcv_df, monkeypatch):
        import finasys as fs

        monkeypatch.setattr(fs, "load", lambda *a, **kw: ohlcv_df)
        from finasys.agents.tools import execute_tool

        result = execute_tool("assess_risk", {"symbol": "TEST"})
        assert "Sharpe" in result

    def test_portfolio_analysis(self, multi_symbol_df, monkeypatch):
        import finasys as fs

        monkeypatch.setattr(fs, "load", lambda *a, **kw: multi_symbol_df)
        from finasys.agents.tools import execute_tool

        result = execute_tool("portfolio_analysis", {"symbols": ["AAPL", "GOOGL"]})
        assert "Portfolio" in result

    def test_portfolio_analysis_with_weights(self, multi_symbol_df, monkeypatch):
        import finasys as fs

        monkeypatch.setattr(fs, "load", lambda *a, **kw: multi_symbol_df)
        from finasys.agents.tools import execute_tool

        result = execute_tool(
            "portfolio_analysis",
            {"symbols": ["AAPL", "GOOGL"], "weights": {"AAPL": 0.7, "GOOGL": 0.3}},
        )
        assert "Portfolio" in result

    def test_screen_stocks(self, ohlcv_df, monkeypatch):
        import finasys as fs

        monkeypatch.setattr(fs, "load", lambda *a, **kw: ohlcv_df)
        from finasys.agents.tools import execute_tool

        result = execute_tool("screen_stocks", {"symbols": ["TEST"], "min_sharpe": -99})
        assert "TEST" in result

    def test_screen_stocks_filters_symbols(self, ohlcv_df, monkeypatch):
        import finasys as fs

        monkeypatch.setattr(fs, "load", lambda *a, **kw: ohlcv_df)
        from finasys.agents.tools import execute_tool

        result = execute_tool(
            "screen_stocks",
            {"symbols": ["TEST"], "min_sharpe": 999, "max_drawdown": 0, "rsi_min": 100, "rsi_max": 0},
        )
        assert result == "[]"

    def test_data_quality_check(self, ohlcv_df, monkeypatch):
        import finasys as fs

        monkeypatch.setattr(fs, "load", lambda *a, **kw: ohlcv_df)
        from finasys.agents.tools import execute_tool

        result = execute_tool("data_quality_check", {"symbol": "TEST"})
        assert "null_counts" in result

    def test_profile_stock(self, ohlcv_df, monkeypatch):
        import finasys as fs

        monkeypatch.setattr(fs, "load", lambda *a, **kw: ohlcv_df)
        from finasys.agents.tools import execute_tool

        result = execute_tool("profile_stock", {"symbol": "TEST"})
        assert "DATA PROFILE" in result


class TestContextEdgeCases:
    """Cover remaining keyword paths in context()."""

    def test_trend_query(self, ohlcv_df):
        from finasys.agents.context import context
        from finasys.features import add_all

        df = add_all(ohlcv_df)
        result = context(df, "What is the trend direction?")
        assert isinstance(result, str)

    def test_compare_query(self, ohlcv_df):
        from finasys.agents.context import context

        result = context(ohlcv_df, "Compare vs other stocks")
        assert isinstance(result, str)

    def test_risk_drawdown_query(self, ohlcv_df):
        from finasys.agents.context import context
        from finasys.features import drawdown

        df = drawdown(ohlcv_df)
        result = context(df, "What is the drawdown risk?")
        assert isinstance(result, str)
        assert "drawdown" in result.lower()

    def test_return_performance_query(self, ohlcv_df):
        from finasys.agents.context import context
        from finasys.features import returns

        df = returns(ohlcv_df, periods=[1, 5])
        result = context(df, "What is the return performance?")
        assert isinstance(result, str)

    def test_bollinger_query(self, ohlcv_df):
        from finasys.agents.context import context
        from finasys.features import bollinger

        df = bollinger(ohlcv_df)
        result = context(df, "Show the bollinger bands")
        assert isinstance(result, str)

    def test_rsi_query(self, ohlcv_df):
        from finasys.agents.context import context
        from finasys.features import rsi

        df = rsi(ohlcv_df)
        result = context(df, "Is it overbought based on RSI?")
        assert isinstance(result, str)

    def test_all_history_query(self, ohlcv_df):
        from finasys.agents.context import context

        result = context(ohlcv_df, "Show the full history")
        assert isinstance(result, str)

    def test_max_tokens_truncation(self, ohlcv_df):
        from finasys.agents.context import context

        result = context(ohlcv_df, "price", max_tokens=20)
        assert len(result) <= 120  # 20 tokens * 4 chars + some slack
