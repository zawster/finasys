"""Extended tests for agents schema and summarize edge cases."""

from datetime import date

import polars as pl


class TestSchemaEdgeCases:
    def test_many_symbols(self):
        """Schema should summarize when > 10 symbols."""
        from finasys.agents.schema import schema

        df = pl.DataFrame(
            {
                "timestamp": [date(2024, 1, 1)] * 15,
                "close": [100.0] * 15,
                "symbol": [f"SYM{i}" for i in range(15)],
            }
        )
        result = schema(df)
        assert "15 unique" in result

    def test_no_nulls(self):
        from finasys.agents.schema import schema

        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "close": [100.0]})
        result = schema(df)
        assert "Null" not in result or "null" not in result.lower()


class TestSummarizeEdgeCases:
    def test_very_short_df(self):
        """Summarize with only 1 row."""
        from finasys.agents.summarize import summarize

        df = pl.DataFrame(
            {
                "timestamp": [date(2024, 1, 1)],
                "close": [100.0],
                "symbol": ["TEST"],
            }
        )
        result = summarize(df)
        assert "TEST" in result
        assert "$100" in result

    def test_no_volume(self):
        """Summarize without volume column."""
        from finasys.agents.summarize import summarize

        df = pl.DataFrame(
            {
                "timestamp": [date(2024, 1, i) for i in range(1, 11)],
                "close": [100.0 + i for i in range(10)],
                "symbol": ["TEST"] * 10,
            }
        )
        result = summarize(df)
        assert "Volume" not in result

    def test_multi_symbol_summary(self):
        from finasys.agents.summarize import summarize

        df = pl.DataFrame(
            {
                "timestamp": [date(2024, 1, 1)] * 2,
                "close": [100.0, 200.0],
                "symbol": ["AAA", "BBB"],
            }
        )
        result = summarize(df)
        assert "AAA" in result or "BBB" in result
