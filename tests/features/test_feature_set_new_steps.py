"""Tests for new FeatureStep subclasses (targets + distributions)."""

import tempfile

from finasys.features.feature_set import (
    ClassifyReturns,
    FeatureSet,
    ForwardReturns,
    RollingKurtosis,
    RollingSkewness,
    TailRatio,
    TripleBarrier,
    VolAdjustedLabels,
    ZscoreReturns,
)


class TestForwardReturnsStep:
    def test_transform(self, ohlcv_df):
        step = ForwardReturns(periods=[1, 5])
        result = step.transform(ohlcv_df)
        assert "fwd_return_1d" in result.columns
        assert "fwd_return_5d" in result.columns

    def test_repr(self):
        step = ForwardReturns(periods=[1, 5])
        assert "ForwardReturns" in repr(step)

    def test_to_dict(self):
        step = ForwardReturns(periods=[1, 5])
        d = step.to_dict()
        assert d["name"] == "ForwardReturns"
        assert d["params"]["periods"] == [1, 5]


class TestClassifyReturnsStep:
    def test_transform(self, ohlcv_df):
        step = ClassifyReturns(period=5)
        result = step.transform(ohlcv_df)
        assert "label_5d" in result.columns

    def test_tuple_thresholds(self, ohlcv_df):
        step = ClassifyReturns(period=5, thresholds=(-0.02, 0.02))
        result = step.transform(ohlcv_df)
        assert "label_5d" in result.columns


class TestTripleBarrierStep:
    def test_transform(self, ohlcv_df):
        step = TripleBarrier(profit_take=0.02, stop_loss=0.02, max_holding=10)
        result = step.transform(ohlcv_df)
        assert "tb_label" in result.columns
        assert "tb_duration" in result.columns
        assert "tb_return" in result.columns


class TestVolAdjustedLabelsStep:
    def test_transform(self, ohlcv_df):
        step = VolAdjustedLabels(period=5)
        result = step.transform(ohlcv_df)
        assert "vol_label_5d" in result.columns


class TestRollingSkewnessStep:
    def test_transform(self, ohlcv_df):
        step = RollingSkewness(window=20)
        result = step.transform(ohlcv_df)
        assert "rolling_skew_20" in result.columns


class TestRollingKurtosisStep:
    def test_transform(self, ohlcv_df):
        step = RollingKurtosis(window=20)
        result = step.transform(ohlcv_df)
        assert "rolling_kurtosis_20" in result.columns


class TestTailRatioStep:
    def test_transform(self, ohlcv_df):
        step = TailRatio(window=30)
        result = step.transform(ohlcv_df)
        assert "tail_ratio_30" in result.columns


class TestZscoreReturnsStep:
    def test_transform(self, ohlcv_df):
        step = ZscoreReturns(window=20)
        result = step.transform(ohlcv_df)
        assert "zscore_returns_20" in result.columns


class TestNewStepsSerialization:
    def test_roundtrip_all_new_steps(self, ohlcv_df):
        """Test that all new steps survive JSON serialize/deserialize."""
        pipeline = FeatureSet(
            [
                ForwardReturns(periods=[1]),
                ClassifyReturns(period=5),
                TripleBarrier(profit_take=0.03, stop_loss=0.03, max_holding=5),
                VolAdjustedLabels(period=5, vol_multiplier=0.5),
                RollingSkewness(window=20),
                RollingKurtosis(window=20),
                TailRatio(window=30, percentile=0.1),
                ZscoreReturns(window=20),
            ]
        )

        # Transform
        result1 = pipeline.transform(ohlcv_df)

        # Save and load
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            pipeline.save(f.name)
            loaded = FeatureSet.load(f.name)

        # Re-transform
        result2 = loaded.transform(ohlcv_df)
        assert result1.shape == result2.shape
        assert result1.columns == result2.columns
