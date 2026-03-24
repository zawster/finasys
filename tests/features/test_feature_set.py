"""Tests for FeatureSet pipeline."""

from finasys.features import MACD, RSI, FeatureSet, Returns


def test_feature_set_transform(ohlcv_df):
    fs = FeatureSet([RSI(period=14), MACD()])
    result = fs.transform(ohlcv_df)

    assert "rsi_14" in result.columns
    assert "macd_line" in result.columns


def test_feature_set_add_chain(ohlcv_df):
    fs = FeatureSet()
    fs.add(RSI()).add(Returns(periods=[1, 5]))
    result = fs.transform(ohlcv_df)

    assert "rsi_14" in result.columns
    assert "returns_1d" in result.columns


def test_feature_set_serialize(tmp_path):
    fs = FeatureSet([RSI(period=14), MACD(), Returns(periods=[1, 5])])
    path = str(tmp_path / "features.json")

    fs.save(path)
    loaded = FeatureSet.load(path)

    assert len(loaded) == 3
    assert loaded.steps[0].name == "RSI"
    assert loaded.steps[0].params["period"] == 14


def test_feature_set_repr():
    fs = FeatureSet([RSI(period=14)])
    r = repr(fs)
    assert "RSI" in r
    assert "14" in r


def test_feature_set_empty():
    fs = FeatureSet()
    assert len(fs) == 0


def test_feature_set_roundtrip(ohlcv_df, tmp_path):
    """Features should produce identical results after save/load."""
    fs = FeatureSet([RSI(period=14), Returns(periods=[1, 5])])
    path = str(tmp_path / "features.json")

    result1 = fs.transform(ohlcv_df)
    fs.save(path)

    fs2 = FeatureSet.load(path)
    result2 = fs2.transform(ohlcv_df)

    assert result1.equals(result2)
