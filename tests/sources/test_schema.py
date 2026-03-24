"""Tests for column schema standardization."""

import polars as pl
import pytest

from finasys.sources.schema import (
    detect_ohlcv_schema,
    standardize_columns,
    validate_ohlcv,
)


def test_standardize_columns_yahoo_format():
    """Standardize Yahoo Finance column names."""
    df = pl.DataFrame(
        {
            "Date": ["2024-01-01"],
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [103.0],
            "Volume": [1000000],
        }
    )
    result = standardize_columns(df)

    assert "timestamp" in result.columns
    assert "open" in result.columns
    assert "close" in result.columns


def test_standardize_columns_adj_close():
    """'Adj Close' maps to 'close'."""
    df = pl.DataFrame(
        {
            "Date": ["2024-01-01"],
            "Adj Close": [103.0],
        }
    )
    result = standardize_columns(df)
    assert "close" in result.columns


def test_detect_ohlcv_schema():
    """Detect whether a DataFrame has OHLCV-like columns."""
    ohlcv = pl.DataFrame(
        {
            "Date": ["2024-01-01"],
            "Close": [100.0],
        }
    )
    assert detect_ohlcv_schema(ohlcv) is True

    not_ohlcv = pl.DataFrame(
        {
            "name": ["Alice"],
            "age": [30],
        }
    )
    assert detect_ohlcv_schema(not_ohlcv) is False


def test_validate_ohlcv_raises_on_missing():
    """validate_ohlcv raises ValueError when required columns are missing."""
    df = pl.DataFrame({"price": [100.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_ohlcv(df)


def test_validate_ohlcv_passes():
    """validate_ohlcv passes for valid DataFrames."""
    df = pl.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "close": [100.0],
        }
    )
    result = validate_ohlcv(df)
    assert result is df
