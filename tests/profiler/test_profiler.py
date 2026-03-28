"""Tests for the smart financial data profiler."""


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
