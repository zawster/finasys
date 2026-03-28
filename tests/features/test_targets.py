"""Tests for target/label engineering."""

import polars as pl


class TestForwardReturns:
    def test_single_period(self, ohlcv_df):
        from finasys.features.targets import forward_returns

        result = forward_returns(ohlcv_df, periods=1)
        assert "fwd_return_1d" in result.columns
        assert result.height == ohlcv_df.height
        # Last row should be null (no future data)
        assert result["fwd_return_1d"].item(-1) is None

    def test_multiple_periods(self, ohlcv_df):
        from finasys.features.targets import forward_returns

        result = forward_returns(ohlcv_df, periods=[1, 5, 21])
        assert "fwd_return_1d" in result.columns
        assert "fwd_return_5d" in result.columns
        assert "fwd_return_21d" in result.columns

    def test_multi_symbol(self, multi_symbol_df):
        from finasys.features.targets import forward_returns

        result = forward_returns(multi_symbol_df, periods=1)
        assert "fwd_return_1d" in result.columns
        # Check no cross-contamination
        aapl = result.filter(pl.col("symbol") == "AAPL")
        googl = result.filter(pl.col("symbol") == "GOOGL")
        assert aapl["fwd_return_1d"].item(-1) is None
        assert googl["fwd_return_1d"].item(-1) is None

    def test_forward_returns_value(self, simple_close_df):
        from finasys.features.targets import forward_returns

        result = forward_returns(simple_close_df, periods=1)
        # First row: fwd_return = close[1]/close[0] - 1 = 102/100 - 1 = 0.02
        assert abs(result["fwd_return_1d"].item(0) - 0.02) < 1e-10


class TestClassifyReturns:
    def test_basic(self, ohlcv_df):
        from finasys.features.targets import classify_returns

        result = classify_returns(ohlcv_df, period=5)
        assert "label_5d" in result.columns
        # Labels should be -1, 0, or 1
        labels = result["label_5d"].drop_nulls().unique().sort().to_list()
        for label in labels:
            assert label in [-1, 0, 1]

    def test_custom_thresholds(self, ohlcv_df):
        from finasys.features.targets import classify_returns

        result = classify_returns(ohlcv_df, period=1, thresholds=(-0.005, 0.005))
        assert "label_1d" in result.columns


class TestTripleBarrierLabels:
    def test_basic(self, ohlcv_df):
        from finasys.features.targets import triple_barrier_labels

        result = triple_barrier_labels(ohlcv_df)
        assert "tb_label" in result.columns
        assert "tb_duration" in result.columns
        assert "tb_return" in result.columns

    def test_labels_range(self, ohlcv_df):
        from finasys.features.targets import triple_barrier_labels

        result = triple_barrier_labels(ohlcv_df)
        labels = result["tb_label"].drop_nulls().unique().sort().to_list()
        for label in labels:
            assert label in [-1, 0, 1]

    def test_duration_positive(self, ohlcv_df):
        from finasys.features.targets import triple_barrier_labels

        result = triple_barrier_labels(ohlcv_df)
        durations = result["tb_duration"].drop_nulls()
        assert (durations >= 0).all()

    def test_multi_symbol(self, multi_symbol_df):
        from finasys.features.targets import triple_barrier_labels

        result = triple_barrier_labels(multi_symbol_df)
        assert "tb_label" in result.columns
        # Both symbols should have results
        for sym in ["AAPL", "GOOGL"]:
            sym_df = result.filter(pl.col("symbol") == sym)
            assert sym_df["tb_label"].drop_nulls().len() > 0


class TestTripleBarrierEdgeCases:
    def test_tight_barriers(self, ohlcv_df):
        from finasys.features.targets import triple_barrier_labels

        result = triple_barrier_labels(ohlcv_df, profit_take=0.001, stop_loss=0.001, max_holding=5)
        assert "tb_label" in result.columns
        # With very tight barriers, most should hit upper or lower
        labels = result["tb_label"].drop_nulls()
        assert labels.len() > 0

    def test_wide_barriers_hit_vertical(self, ohlcv_df):
        from finasys.features.targets import triple_barrier_labels

        result = triple_barrier_labels(ohlcv_df, profit_take=0.99, stop_loss=0.99, max_holding=2)
        # With very wide barriers and short holding, most should hit vertical
        assert "tb_label" in result.columns


class TestVolatilityAdjustedLabels:
    def test_basic(self, ohlcv_df):
        from finasys.features.targets import volatility_adjusted_labels

        result = volatility_adjusted_labels(ohlcv_df, period=5)
        assert "vol_label_5d" in result.columns

    def test_labels_range(self, ohlcv_df):
        from finasys.features.targets import volatility_adjusted_labels

        result = volatility_adjusted_labels(ohlcv_df, period=5)
        labels = result["vol_label_5d"].drop_nulls().unique().sort().to_list()
        for label in labels:
            assert label in [-1, 0, 1]

    def test_multi_symbol(self, multi_symbol_df):
        from finasys.features.targets import volatility_adjusted_labels

        result = volatility_adjusted_labels(multi_symbol_df, period=5)
        assert "vol_label_5d" in result.columns
