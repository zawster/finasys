"""Tests for the load() dispatcher — ticker routing and cache integration."""

from datetime import date
from unittest.mock import patch

import polars as pl


def _mock_yahoo_df(n=30, symbol="TEST"):
    """Create a Polars DataFrame like fetch_yahoo returns."""
    return pl.DataFrame(
        {
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 6, 1), eager=True)[:n],
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [101.0 + i * 0.1 for i in range(n)],
            "low": [99.0 + i * 0.1 for i in range(n)],
            "close": [100.5 + i * 0.1 for i in range(n)],
            "volume": [1e6] * n,
            "symbol": [symbol.upper()] * n,
        }
    )


class TestSingleTickerWithCache:
    def test_cache_miss_then_hit(self, monkeypatch):
        from finasys.sources import _load_single_ticker
        from finasys.sources import cache as cache_mod

        # Clear cache
        cache_mod.cache_clear("CTEST")

        # Mock fetch_yahoo
        mock_df = _mock_yahoo_df(20, "CTEST")
        with patch("finasys.sources.fetch_yahoo", return_value=mock_df) as mock_fetch:
            # First call — cache miss
            result = _load_single_ticker("CTEST", start="2024-01-01", end=None, interval="1d", use_cache=True)
            assert result.height == 20
            assert mock_fetch.call_count == 1

        # Second call — cache hit (no fetch)
        with patch("finasys.sources.fetch_yahoo"):
            result2 = _load_single_ticker("CTEST", start="2024-01-01", end=None, interval="1d", use_cache=True)
            # Should NOT call fetch_yahoo because cache has data
            assert result2.height > 0

        cache_mod.cache_clear("CTEST")

    def test_no_cache(self, monkeypatch):
        from finasys.sources import _load_single_ticker

        mock_df = _mock_yahoo_df(15, "NOCACHE")
        with patch("finasys.sources.fetch_yahoo", return_value=mock_df):
            result = _load_single_ticker("NOCACHE", start=None, end=None, interval="1d", use_cache=False)
            assert result.height == 15


class TestMultiTickerWithCache:
    def test_multi_no_cache(self):
        from finasys.sources import _load_multi_ticker

        mock_df = pl.concat([_mock_yahoo_df(10, "AA"), _mock_yahoo_df(10, "BB")]).sort(["timestamp", "symbol"])
        with patch("finasys.sources.fetch_yahoo_multi", return_value=mock_df):
            result = _load_multi_ticker(["AA", "BB"], start=None, end=None, interval="1d", use_cache=False)
            assert result.height == 20

    def test_multi_with_cache(self):
        from finasys.sources import _load_multi_ticker
        from finasys.sources import cache as cache_mod

        cache_mod.cache_clear("MA")
        cache_mod.cache_clear("MB")

        mock_df = pl.concat([_mock_yahoo_df(10, "MA"), _mock_yahoo_df(10, "MB")]).sort(["timestamp", "symbol"])
        with patch("finasys.sources.fetch_yahoo_multi", return_value=mock_df):
            result = _load_multi_ticker(["MA", "MB"], start=None, end=None, interval="1d", use_cache=True)
            assert result.height == 20

        cache_mod.cache_clear("MA")
        cache_mod.cache_clear("MB")

    def test_multi_partial_cache(self):
        """One symbol cached, one needs fetching."""
        from finasys.sources import _load_multi_ticker
        from finasys.sources import cache as cache_mod

        # Pre-cache one symbol
        cache_mod.cache_clear("PC")
        cache_mod.cache_clear("PD")
        cache_mod.cache_put(_mock_yahoo_df(10, "PC"), "PC")

        fresh_df = _mock_yahoo_df(10, "PD")
        with patch("finasys.sources.fetch_yahoo_multi", return_value=fresh_df):
            result = _load_multi_ticker(["PC", "PD"], start=None, end=None, interval="1d", use_cache=True)
            # Should have data from both symbols
            assert result.height > 0
            symbols = result["symbol"].unique().to_list()
            assert "PD" in symbols

        cache_mod.cache_clear("PC")
        cache_mod.cache_clear("PD")
