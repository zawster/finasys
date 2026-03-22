"""Tests for the ak.load() dispatcher and local file loading."""

from pathlib import Path

import polars as pl
import pytest


def test_load_local_csv(ohlcv_df: pl.DataFrame, tmp_path: Path):
    """load() with a CSV file path returns standardized DataFrame."""
    csv_path = tmp_path / "test_data.csv"
    ohlcv_df.write_csv(csv_path)

    from alphakit.sources import load

    result = load(str(csv_path))

    assert isinstance(result, pl.DataFrame)
    assert "close" in result.columns
    assert result.height == ohlcv_df.height


def test_load_local_parquet(ohlcv_df: pl.DataFrame, tmp_path: Path):
    """load() with a Parquet file path returns standardized DataFrame."""
    pq_path = tmp_path / "test_data.parquet"
    ohlcv_df.write_parquet(pq_path)

    from alphakit.sources import load

    result = load(str(pq_path))

    assert isinstance(result, pl.DataFrame)
    assert "close" in result.columns


def test_load_local_file_not_found():
    """load() raises FileNotFoundError for missing files."""
    from alphakit.sources import load

    with pytest.raises(FileNotFoundError):
        load("./nonexistent_file.csv")


def test_load_local_unsupported_format(tmp_path: Path):
    """load() raises ValueError for unsupported file formats."""
    bad_file = tmp_path / "data.xlsx"
    bad_file.write_text("test")

    from alphakit.sources import load

    with pytest.raises(ValueError, match="Unsupported file format"):
        load(str(bad_file))


def test_load_detects_file_vs_ticker():
    """_is_file_path correctly distinguishes files from tickers."""
    from alphakit.sources import _is_file_path

    assert _is_file_path("./data.csv") is True
    assert _is_file_path("data.parquet") is True
    assert _is_file_path("/home/user/data.csv") is True
    assert _is_file_path("C:\\data\\file.csv") is True
    assert _is_file_path("AAPL") is False
    assert _is_file_path("GOOGL") is False
    assert _is_file_path("BRK.B") is False  # Dot in ticker, but not a file extension


def test_load_pandas_backend(ohlcv_df: pl.DataFrame, tmp_path: Path):
    """load() with backend='pandas' returns a pandas DataFrame."""
    csv_path = tmp_path / "test.csv"
    ohlcv_df.write_csv(csv_path)

    from alphakit.sources import load

    result = load(str(csv_path), backend="pandas")

    import pandas as pd

    assert isinstance(result, pd.DataFrame)
