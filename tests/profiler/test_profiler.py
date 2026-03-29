"""Tests for the smart financial data profiler."""

from datetime import date

import polars as pl


class TestProfile:
    def test_basic(self, ohlcv_df):
        from finasys.profiler import profile

        report = profile(ohlcv_df)
        assert report.shape == (ohlcv_df.height, ohlcv_df.width)
        assert len(report.date_range) == 2
        assert report.symbols == ["AAPL"]

    def test_column_stats(self, ohlcv_df):
        from finasys.profiler import profile

        report = profile(ohlcv_df)
        assert "close" in report.column_stats
        cs = report.column_stats["close"]
        assert cs.count == ohlcv_df.height
        assert cs.null_count == 0
        assert cs.mean is not None
        assert cs.std is not None
        assert cs.min is not None
        assert cs.max is not None

    def test_quantiles(self, ohlcv_df):
        from finasys.profiler import profile

        report = profile(ohlcv_df)
        cs = report.column_stats["close"]
        assert len(cs.quantiles) > 0
        assert "0.5" in cs.quantiles

    def test_quality_report(self, ohlcv_df):
        from finasys.profiler import profile

        report = profile(ohlcv_df)
        q = report.quality
        assert isinstance(q.duplicate_rows, int)
        assert isinstance(q.zero_volume_days, int)

    def test_distribution_report(self, ohlcv_df):
        from finasys.profiler import profile

        report = profile(ohlcv_df)
        d = report.distribution
        assert isinstance(d.returns_skewness, float)
        assert isinstance(d.returns_kurtosis, float)
        assert isinstance(d.jarque_bera_stat, float)
        assert d.jarque_bera_stat >= 0

    def test_to_dict(self, ohlcv_df):
        from finasys.profiler import profile

        report = profile(ohlcv_df)
        d = report.to_dict()
        assert "shape" in d
        assert "column_stats" in d
        assert "quality" in d
        assert "distribution" in d

    def test_multi_symbol(self, multi_symbol_df):
        from finasys.profiler import profile

        report = profile(multi_symbol_df)
        assert "AAPL" in report.symbols
        assert "GOOGL" in report.symbols

    def test_missing_dates_detected(self, ohlcv_df):
        from finasys.profiler import profile

        # Remove some rows to create gaps
        df = ohlcv_df.head(50).filter(pl.col("timestamp").cast(str) != str(ohlcv_df["timestamp"].item(5)))
        report = profile(df)
        assert len(report.quality.missing_dates) > 0

    def test_no_timestamp_column(self):
        from finasys.profiler import profile

        df = pl.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
        report = profile(df)
        assert report.date_range == ("", "")
        assert report.quality.missing_dates == []

    def test_no_volume_column(self):
        from finasys.profiler import profile

        df = pl.DataFrame(
            {
                "timestamp": [date(2024, 1, i) for i in range(1, 11)],
                "close": [100.0 + i for i in range(10)],
            }
        )
        report = profile(df)
        assert report.quality.zero_volume_days == 0

    def test_non_numeric_column(self):
        from finasys.profiler import profile

        df = pl.DataFrame(
            {
                "timestamp": [date(2024, 1, i) for i in range(1, 6)],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "notes": ["a", "b", "c", "d", "e"],
            }
        )
        report = profile(df)
        cs = report.column_stats["notes"]
        assert cs.mean is None
        assert cs.quantiles == {}

    def test_missing_close_column(self):
        from finasys.profiler import profile

        df = pl.DataFrame({"price": [100.0, 101.0, 102.0, 103.0, 104.0]})
        report = profile(df, column="nonexistent")
        assert report.distribution.returns_skewness == 0.0

    def test_very_short_df(self):
        from finasys.profiler import profile

        df = pl.DataFrame(
            {
                "timestamp": [date(2024, 1, 1)],
                "close": [100.0],
            }
        )
        report = profile(df)
        assert report.shape == (1, 2)

    def test_zero_volume_detected(self):
        from finasys.profiler import profile

        df = pl.DataFrame(
            {
                "timestamp": [date(2024, 1, i) for i in range(1, 11)],
                "close": [100.0 + i for i in range(10)],
                "volume": [1000.0, 0.0, 1000.0, 0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
            }
        )
        report = profile(df)
        assert report.quality.zero_volume_days == 2

    def test_duplicates_detected(self):
        from finasys.profiler import profile

        df = pl.DataFrame(
            {
                "timestamp": [date(2024, 1, 1)] * 3,
                "close": [100.0] * 3,
            }
        )
        report = profile(df)
        assert report.quality.duplicate_rows > 0


class TestProfileEdgeCases:
    def test_short_distribution(self):
        from finasys.profiler import profile

        df = pl.DataFrame({"close": [100.0, 101.0, 102.0]})
        report = profile(df)
        # Too few returns for distribution analysis
        assert report.distribution.jarque_bera_stat == 0.0

    def test_constant_price_distribution(self):
        from finasys.profiler import profile

        df = pl.DataFrame({"close": [100.0] * 20})
        report = profile(df)
        # Zero variance returns
        assert report.distribution.returns_skewness == 0.0

    def test_column_with_few_values(self):
        from finasys.profiler import profile

        df = pl.DataFrame({"close": [100.0, None, None]})
        report = profile(df)
        cs = report.column_stats["close"]
        assert cs.null_count == 2

    def test_no_outliers_in_stable_data(self):
        from finasys.profiler import profile

        # Very stable prices -> no outliers
        df = pl.DataFrame(
            {
                "timestamp": [date(2024, 1, i) for i in range(1, 21)],
                "close": [100.0 + i * 0.01 for i in range(20)],
                "open": [100.0 + i * 0.01 for i in range(20)],
                "high": [100.0 + i * 0.01 + 0.005 for i in range(20)],
                "low": [100.0 + i * 0.01 - 0.005 for i in range(20)],
            }
        )
        report = profile(df)
        assert report.quality.price_outliers == {}

    def test_empty_timestamp(self):
        from finasys.profiler import profile

        df = pl.DataFrame(
            {
                "timestamp": pl.Series([], dtype=pl.Date),
                "close": pl.Series([], dtype=pl.Float64),
            }
        )
        report = profile(df)
        assert report.shape == (0, 2)
        assert report.quality.missing_dates == []


class TestProfileSummary:
    def test_basic(self, ohlcv_df):
        from finasys.profiler import profile_summary

        result = profile_summary(ohlcv_df)
        assert isinstance(result, str)
        assert "DATA PROFILE" in result
        assert "Date range" in result

    def test_contains_distribution_info(self, ohlcv_df):
        from finasys.profiler import profile_summary

        result = profile_summary(ohlcv_df)
        assert "skew=" in result
        assert "kurtosis=" in result

    def test_simple_df(self, simple_close_df):
        from finasys.profiler import profile_summary

        result = profile_summary(simple_close_df)
        assert isinstance(result, str)

    def test_no_issues_message(self):
        from finasys.profiler import profile_summary

        # Create a clean df with no gaps, no outliers, consecutive business days
        dates = []
        d = date(2024, 1, 2)
        from datetime import timedelta

        while len(dates) < 20:
            if d.weekday() < 5:
                dates.append(d)
            d += timedelta(days=1)

        df = pl.DataFrame(
            {
                "timestamp": dates,
                "close": [100.0 + i * 0.1 for i in range(20)],
            }
        )
        result = profile_summary(df)
        assert "No issues detected" in result

    def test_with_symbols(self, multi_symbol_df):
        from finasys.profiler import profile_summary

        result = profile_summary(multi_symbol_df)
        assert "Symbols:" in result

    def test_quality_issues_shown_with_all_issues(self):
        from finasys.profiler import profile_summary

        # Create data that triggers all quality issue branches
        dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 3), date(2024, 1, 5), date(2024, 1, 8)]
        df = pl.DataFrame(
            {
                "timestamp": dates,
                "close": [100.0, 101.0, 101.0, 130.0, 103.0],  # >20% jump = suspected split, >4sigma = outlier
                "volume": [1000.0, 0.0, 1000.0, 1000.0, 1000.0],
            }
        )
        result = profile_summary(df)
        # Should hit duplicate, zero-volume, suspected split branches
        assert "Quality issues:" in result
