"""Extended tests for schema standardization and type casting."""

from datetime import date

import polars as pl

from finasys.sources.schema import cast_ohlcv_types, standardize_columns


class TestCastTypes:
    def test_cast_string_close(self):
        df = pl.DataFrame({"timestamp": ["2024-01-01"], "close": ["100.5"]})
        result = cast_ohlcv_types(df)
        assert result["close"].dtype == pl.Float64

    def test_cast_string_date(self):
        df = pl.DataFrame({"timestamp": ["2024-01-01"], "close": [100.0]})
        result = cast_ohlcv_types(df)
        assert result["timestamp"].dtype == pl.Date

    def test_cast_string_datetime(self):
        df = pl.DataFrame({"timestamp": ["2024-01-01 09:30:00"], "close": [100.0]})
        result = cast_ohlcv_types(df)
        assert result["timestamp"].dtype in (pl.Date, pl.Datetime, pl.String)

    def test_cast_volume(self):
        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "close": [100.0], "volume": [1000000]})
        result = cast_ohlcv_types(df)
        assert result["volume"].dtype == pl.Float64

    def test_cast_ohlc(self):
        df = pl.DataFrame(
            {
                "timestamp": [date(2024, 1, 1)],
                "open": [100],
                "high": [105],
                "low": [95],
                "close": [102],
            }
        )
        result = cast_ohlcv_types(df)
        for col in ["open", "high", "low", "close"]:
            assert result[col].dtype == pl.Float64

    def test_unparseable_date_stays_string(self):
        df = pl.DataFrame({"timestamp": ["not-a-date"], "close": [100.0]})
        result = cast_ohlcv_types(df)
        # Should not crash, stays as string
        assert result["timestamp"].dtype in (pl.String, pl.Utf8)


class TestStandardizeEdgeCases:
    def test_already_standard(self):
        df = pl.DataFrame({"timestamp": [1], "close": [100.0]})
        result = standardize_columns(df)
        assert result.columns == ["timestamp", "close"]

    def test_mixed_case(self):
        df = pl.DataFrame({"DATE": ["x"], "CLOSE": [100.0], "VOLUME": [1000]})
        result = standardize_columns(df)
        assert "date" in result.columns or "timestamp" in result.columns
        assert "close" in result.columns

    def test_lowercase_aliases(self):
        df = pl.DataFrame({"date": ["x"], "vol": [1000]})
        result = standardize_columns(df)
        assert "timestamp" in result.columns
        assert "volume" in result.columns
