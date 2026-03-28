"""Tests for distribution analysis features."""


class TestRollingSkewness:
    def test_basic(self, ohlcv_df):
        from finasys.features.distributions import rolling_skewness

        result = rolling_skewness(ohlcv_df, window=20)
        assert "rolling_skew_20" in result.columns
        assert result.height == ohlcv_df.height

    def test_multi_symbol(self, multi_symbol_df):
        from finasys.features.distributions import rolling_skewness

        result = rolling_skewness(multi_symbol_df, window=20)
        assert "rolling_skew_20" in result.columns


class TestRollingKurtosis:
    def test_basic(self, ohlcv_df):
        from finasys.features.distributions import rolling_kurtosis

        result = rolling_kurtosis(ohlcv_df, window=20)
        assert "rolling_kurtosis_20" in result.columns
        assert result.height == ohlcv_df.height
        # First window-1 values should be null/NaN
        first_vals = result["rolling_kurtosis_20"].head(19)
        assert first_vals.is_null().all() or first_vals.is_nan().all()

    def test_multi_symbol(self, multi_symbol_df):
        from finasys.features.distributions import rolling_kurtosis

        result = rolling_kurtosis(multi_symbol_df, window=20)
        assert "rolling_kurtosis_20" in result.columns


class TestTailRatio:
    def test_basic(self, ohlcv_df):
        from finasys.features.distributions import tail_ratio

        result = tail_ratio(ohlcv_df, window=30)
        assert "tail_ratio_30" in result.columns
        assert result.height == ohlcv_df.height

    def test_multi_symbol(self, multi_symbol_df):
        from finasys.features.distributions import tail_ratio

        result = tail_ratio(multi_symbol_df, window=30)
        assert "tail_ratio_30" in result.columns


class TestRollingJarqueBera:
    def test_basic(self, ohlcv_df):
        from finasys.features.distributions import rolling_jarque_bera

        result = rolling_jarque_bera(ohlcv_df, window=30)
        assert "rolling_jb_30" in result.columns
        # JB statistic should be non-negative
        jb_vals = result["rolling_jb_30"].drop_nulls()
        non_nan = jb_vals.filter(~jb_vals.is_nan())
        if non_nan.len() > 0:
            assert (non_nan >= 0).all()


class TestZscoreReturns:
    def test_basic(self, ohlcv_df):
        from finasys.features.distributions import zscore_returns

        result = zscore_returns(ohlcv_df, window=20)
        assert "zscore_returns_20" in result.columns
        assert result.height == ohlcv_df.height

    def test_multi_symbol(self, multi_symbol_df):
        from finasys.features.distributions import zscore_returns

        result = zscore_returns(multi_symbol_df, window=20)
        assert "zscore_returns_20" in result.columns
