"""Tests for Yahoo Finance source with mocked yfinance."""

from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest


def _make_mock_history(n=50):
    """Create a mock pandas DataFrame like yfinance returns."""
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Open": [100.0 + i * 0.1 for i in range(n)],
            "High": [101.0 + i * 0.1 for i in range(n)],
            "Low": [99.0 + i * 0.1 for i in range(n)],
            "Close": [100.5 + i * 0.1 for i in range(n)],
            "Volume": [1_000_000] * n,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


class TestFetchYahoo:
    def test_single_symbol(self):
        from finasys.sources.yahoo import fetch_yahoo

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_mock_history()

        with patch("finasys.sources.yahoo.yf.Ticker", return_value=mock_ticker):
            df = fetch_yahoo("TEST")

        assert isinstance(df, pl.DataFrame)
        assert "close" in df.columns
        assert "timestamp" in df.columns
        assert "symbol" in df.columns
        assert df["symbol"][0] == "TEST"
        assert df.height == 50

    def test_empty_data_raises(self):
        from finasys.sources.yahoo import fetch_yahoo

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("finasys.sources.yahoo.yf.Ticker", return_value=mock_ticker):
            with pytest.raises(ValueError, match="No data"):
                fetch_yahoo("INVALID")

    def test_custom_dates(self):
        from finasys.sources.yahoo import fetch_yahoo

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_mock_history(20)

        with patch("finasys.sources.yahoo.yf.Ticker", return_value=mock_ticker):
            df = fetch_yahoo("TEST", start="2024-01-01", end="2024-02-01")

        assert df.height == 20
        mock_ticker.history.assert_called_once()


class TestFetchYahooMulti:
    def test_multi_symbols(self):
        from finasys.sources.yahoo import fetch_yahoo_multi

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_mock_history(30)

        with patch("finasys.sources.yahoo.yf.Ticker", return_value=mock_ticker):
            df = fetch_yahoo_multi(["AAA", "BBB"])

        assert df.height == 60
        assert "symbol" in df.columns

    def test_all_fail_raises(self):
        from finasys.sources.yahoo import fetch_yahoo_multi

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("finasys.sources.yahoo.yf.Ticker", return_value=mock_ticker):
            with pytest.raises(ValueError, match="No data"):
                fetch_yahoo_multi(["BAD1", "BAD2"])

    def test_partial_fail(self):
        from finasys.sources.yahoo import fetch_yahoo_multi

        call_count = [0]

        def mock_history(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _make_mock_history(20)
            return pd.DataFrame()

        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = mock_history

        with patch("finasys.sources.yahoo.yf.Ticker", return_value=mock_ticker):
            df = fetch_yahoo_multi(["GOOD", "BAD"])

        assert df.height == 20


class TestLoadDispatch:
    """Test the load() dispatcher with mocked Yahoo."""

    def test_load_single_ticker(self):
        from finasys.sources import load

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_mock_history(30)

        with patch("finasys.sources.yahoo.yf.Ticker", return_value=mock_ticker):
            df = load("TEST", start="2024-01-01", use_cache=False)

        assert isinstance(df, pl.DataFrame)
        assert df.height == 30

    def test_load_list_of_tickers(self):
        from finasys.sources import load

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = _make_mock_history(20)

        with patch("finasys.sources.yahoo.yf.Ticker", return_value=mock_ticker):
            df = load(["AAA", "BBB"], start="2024-01-01", use_cache=False)

        assert df.height == 40
