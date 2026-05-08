"""Tests for financial data quality checks."""

from datetime import date

import polars as pl
import pytest


def test_detect_gaps_single_symbol():
    from finasys.quality import detect_gaps

    df = pl.DataFrame(
        {
            "timestamp": [date(2024, 1, 1), date(2024, 1, 3)],
            "close": [100.0, 101.0],
        }
    )

    assert detect_gaps(df) == ["2024-01-02"]


def test_detect_gaps_multi_symbol(multi_symbol_df):
    from finasys.quality import detect_gaps

    result = detect_gaps(multi_symbol_df)

    assert set(result) == {"AAPL", "GOOGL"}


def test_detect_gaps_invalid_frequency(ohlcv_df):
    from finasys.quality import detect_gaps

    with pytest.raises(ValueError, match="1bd"):
        detect_gaps(ohlcv_df, freq="1d")


def test_flag_outliers_zscore(ohlcv_df):
    from finasys.quality import flag_outliers

    df = ohlcv_df.with_columns(
        pl.when(pl.arange(0, pl.len()) == 50).then(1000.0).otherwise(pl.col("close")).alias("close")
    )
    result = flag_outliers(df, columns=["close"], threshold=3.0)

    assert "close_outlier" in result.columns
    assert result["close_outlier"].sum() >= 1


def test_flag_outliers_iqr(ohlcv_df):
    from finasys.quality import flag_outliers

    result = flag_outliers(ohlcv_df, method="iqr", columns=["close"])

    assert "close_outlier" in result.columns


def test_detect_splits(ohlcv_df):
    from finasys.quality import detect_splits

    df = ohlcv_df.with_columns(
        pl.when(pl.arange(0, pl.len()) == 20).then(pl.col("close") * 2).otherwise(pl.col("close")).alias("close")
    )
    result = detect_splits(df, threshold=0.3)

    assert "suspected_split" in result.columns
    assert result["suspected_split"].sum() >= 1


def test_completeness_report(ohlcv_df):
    from finasys.quality import completeness_report

    result = completeness_report(ohlcv_df)

    assert result["rows"] == ohlcv_df.height
    assert "null_counts" in result
    assert "date_gap_count" in result
