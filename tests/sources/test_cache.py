"""Tests for the DuckDB cache layer."""

from datetime import date
from pathlib import Path

import duckdb
import polars as pl
import pytest

from finasys.sources.cache import cache_clear, cache_get, cache_put


@pytest.fixture
def cache_df():
    """Clean DataFrame for cache tests."""
    return pl.DataFrame(
        {
            "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 4, 9), eager=True)[:100],
            "open": [100.0 + i * 0.1 for i in range(100)],
            "high": [101.0 + i * 0.1 for i in range(100)],
            "low": [99.0 + i * 0.1 for i in range(100)],
            "close": [100.5 + i * 0.1 for i in range(100)],
            "volume": [1e6] * 100,
        }
    )


class TestCachePutGet:
    def test_roundtrip(self, cache_df):
        cache_clear("CTEST1")
        cache_put(cache_df, "CTEST1")
        result = cache_get("CTEST1")
        assert result is not None
        assert result.height == 100
        cache_clear("CTEST1")

    def test_get_nonexistent(self):
        cache_clear("NONEXIST")
        assert cache_get("NONEXIST") is None

    def test_clear_specific(self, cache_df):
        cache_put(cache_df, "CTEST2")
        cache_clear("CTEST2")
        assert cache_get("CTEST2") is None

    def test_clear_all(self, cache_df):
        cache_put(cache_df, "CTEST3")
        cache_clear()
        assert cache_get("CTEST3") is None

    def test_date_range_filter(self, cache_df):
        cache_clear("CTEST4")
        cache_put(cache_df, "CTEST4")
        result = cache_get("CTEST4", start="2024-02-01", end="2024-03-01")
        assert result is not None
        assert result.height < 100
        cache_clear("CTEST4")

    def test_empty_df_skipped(self):
        empty = pl.DataFrame({"timestamp": [], "close": []}).cast({"close": pl.Float64})
        cache_clear("CEMPTY")
        cache_put(empty, "CEMPTY")
        assert cache_get("CEMPTY") is None

    def test_cache_disabled(self, cache_df, monkeypatch):
        from finasys.utils.config import config

        monkeypatch.setattr(config, "cache_enabled", False)
        cache_put(cache_df, "DISABLED")
        assert cache_get("DISABLED") is None
        monkeypatch.setattr(config, "cache_enabled", True)

    def test_symbol_override(self, cache_df):
        """cache_put should use the passed symbol name, not DataFrame's symbol column."""
        df_with_sym = cache_df.with_columns(pl.lit("WRONG").alias("symbol"))
        cache_clear("CORRECT")
        cache_put(df_with_sym, "CORRECT")
        result = cache_get("CORRECT")
        assert result is not None
        assert result["symbol"][0] == "CORRECT"
        cache_clear("CORRECT")

    def test_connection_fallback_path(self, tmp_path, monkeypatch):
        import finasys.sources.cache as cache_mod

        real_connect = duckdb.connect
        calls = []

        def fake_connect(path):
            calls.append(Path(path))
            if len(calls) == 1:
                raise duckdb.IOException("primary cache unavailable")
            return real_connect(str(tmp_path / "fallback.duckdb"))

        monkeypatch.setattr(cache_mod.duckdb, "connect", fake_connect)
        monkeypatch.setattr(cache_mod.tempfile, "gettempdir", lambda: str(tmp_path))

        conn = cache_mod._get_connection()
        conn.close()

        assert len(calls) == 2

    def test_cache_get_connection_failure_returns_none(self, monkeypatch):
        import finasys.sources.cache as cache_mod

        monkeypatch.setattr(cache_mod, "_get_connection", lambda: (_ for _ in ()).throw(RuntimeError("no db")))

        assert cache_mod.cache_get("FAIL") is None

    def test_cache_put_connection_failure_is_ignored(self, cache_df, monkeypatch):
        import finasys.sources.cache as cache_mod

        monkeypatch.setattr(cache_mod, "_get_connection", lambda: (_ for _ in ()).throw(RuntimeError("no db")))

        cache_mod.cache_put(cache_df, "FAIL")

    def test_cache_clear_connection_failure_is_ignored(self, monkeypatch):
        import finasys.sources.cache as cache_mod

        monkeypatch.setattr(cache_mod, "_get_connection", lambda: (_ for _ in ()).throw(RuntimeError("no db")))

        cache_mod.cache_clear("FAIL")

    def test_cache_get_query_failure_returns_none(self, monkeypatch):
        import finasys.sources.cache as cache_mod

        class BadConnection:
            def execute(self, *args, **kwargs):
                raise RuntimeError("query failed")

            def close(self):
                self.closed = True

        monkeypatch.setattr(cache_mod, "_get_connection", lambda: BadConnection())

        assert cache_mod.cache_get("FAIL") is None

    def test_cache_put_insert_failure_is_ignored(self, cache_df, monkeypatch):
        import finasys.sources.cache as cache_mod

        class BadConnection:
            def execute(self, *args, **kwargs):
                raise RuntimeError("insert failed")

            def close(self):
                self.closed = True

        monkeypatch.setattr(cache_mod, "_get_connection", lambda: BadConnection())

        cache_mod.cache_put(cache_df, "FAIL")
