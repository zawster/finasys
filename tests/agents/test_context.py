"""Tests for agent context extraction."""

from finasys.agents import context, schema


class TestContext:
    def test_context_returns_string(self, ohlcv_df):
        result = context(ohlcv_df, "What is the current price?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_context_recent_query(self, ohlcv_df):
        result = context(ohlcv_df, "What is the latest price?")
        assert isinstance(result, str)

    def test_context_json_format(self, ohlcv_df):
        result = context(ohlcv_df, "current price", format="json")
        import json

        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_context_text_format(self, ohlcv_df):
        result = context(ohlcv_df, "current price", format="text")
        assert isinstance(result, str)


class TestSchema:
    def test_schema_basic(self, ohlcv_df):
        result = schema(ohlcv_df)
        assert "100 rows" in result or "100" in result
        assert "columns" in result.lower()

    def test_schema_includes_symbol(self, ohlcv_df):
        result = schema(ohlcv_df)
        assert "AAPL" in result

    def test_schema_includes_columns(self, ohlcv_df):
        result = schema(ohlcv_df)
        assert "close" in result
        assert "timestamp" in result
