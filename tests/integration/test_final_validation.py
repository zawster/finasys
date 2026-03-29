"""
FINAL VALIDATION TEST SUITE
============================
Comprehensive end-to-end testing of every public function in finasys.
Written from the perspective of a neutral end user who just ran:
    pip install finasys

Tests every function, every parameter variant, edge cases, and
cross-module integration.
"""

import json
import os
import tempfile
from datetime import date

import numpy as np
import polars as pl
import pytest

# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def sample_ohlcv() -> pl.DataFrame:
    """100-row synthetic OHLCV with realistic price behavior."""
    rng = np.random.RandomState(42)
    n = 100
    dates = pl.date_range(date(2024, 1, 1), date(2024, 5, 15), eager=True)[:n]
    close = np.cumsum(rng.normal(0.2, 2.0, n)) + 150
    high = close + rng.uniform(0.5, 3.0, n)
    low = close - rng.uniform(0.5, 3.0, n)
    open_ = low + rng.uniform(0.2, 0.8, n) * (high - low)
    volume = rng.uniform(1e6, 1e7, n)

    return pl.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": ["TEST"] * n,
        }
    )


@pytest.fixture
def multi_sym() -> pl.DataFrame:
    """Two-symbol DataFrame for cross-sectional tests."""
    rng = np.random.RandomState(99)
    n = 60
    dates = pl.date_range(date(2024, 1, 1), date(2024, 3, 31), eager=True)[:n]

    frames = []
    for sym, base_price in [("AAA", 100.0), ("BBB", 200.0)]:
        close = np.cumsum(rng.normal(0.1, 1.5, n)) + base_price
        high = close + rng.uniform(0.3, 2.0, n)
        low = close - rng.uniform(0.3, 2.0, n)
        open_ = low + rng.uniform(0.2, 0.8, n) * (high - low)
        volume = rng.uniform(5e5, 5e6, n)
        frames.append(
            pl.DataFrame(
                {
                    "timestamp": dates,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "symbol": [sym] * n,
                }
            )
        )

    return pl.concat(frames).sort(["timestamp", "symbol"])


@pytest.fixture
def minimal_close() -> pl.DataFrame:
    """Bare minimum: just timestamp + close, no symbol."""
    return pl.DataFrame(
        {
            "timestamp": [date(2024, 1, i) for i in range(1, 21)],
            "close": [100 + i * 0.5 + ((-1) ** i) * 0.3 for i in range(20)],
        }
    )


# ============================================================
# 1. SOURCES MODULE
# ============================================================


class TestLoad:
    """Test fs.load() -- the single entry point."""

    def test_load_csv(self, sample_ohlcv, tmp_path):
        sample_ohlcv.write_csv(tmp_path / "data.csv")
        import finasys as fs

        df = fs.load(str(tmp_path / "data.csv"))
        assert isinstance(df, pl.DataFrame)
        assert df.height == 100
        assert "close" in df.columns

    def test_load_parquet(self, sample_ohlcv, tmp_path):
        sample_ohlcv.write_parquet(tmp_path / "data.parquet")
        import finasys as fs

        df = fs.load(str(tmp_path / "data.parquet"))
        assert df.height == 100

    def test_load_json_file(self, sample_ohlcv, tmp_path):
        # JSON without symbol/date complexity
        simple = pl.DataFrame(
            {
                "timestamp": ["2024-01-01", "2024-01-02"],
                "close": [100.0, 101.0],
            }
        )
        simple.write_json(tmp_path / "data.json")
        import finasys as fs

        df = fs.load(str(tmp_path / "data.json"))
        assert "close" in df.columns

    def test_load_pandas_backend(self, sample_ohlcv, tmp_path):
        sample_ohlcv.write_csv(tmp_path / "data.csv")
        import pandas as pd

        import finasys as fs

        df = fs.load(str(tmp_path / "data.csv"), backend="pandas")
        assert isinstance(df, pd.DataFrame)

    def test_load_polars_backend_explicit(self, sample_ohlcv, tmp_path):
        sample_ohlcv.write_csv(tmp_path / "data.csv")
        import finasys as fs

        df = fs.load(str(tmp_path / "data.csv"), backend="polars")
        assert isinstance(df, pl.DataFrame)

    def test_load_file_not_found(self):
        import finasys as fs

        with pytest.raises(FileNotFoundError):
            fs.load("./does_not_exist.csv")

    def test_load_unsupported_format(self, tmp_path):
        (tmp_path / "bad.xlsx").write_text("x")
        import finasys as fs

        with pytest.raises(ValueError, match="Unsupported"):
            fs.load(str(tmp_path / "bad.xlsx"))

    def test_file_path_detection(self):
        from finasys.sources import _is_file_path

        # Files
        assert _is_file_path("data.csv") is True
        assert _is_file_path("./path/to/data.parquet") is True
        assert _is_file_path("C:\\Users\\data.csv") is True
        assert _is_file_path("/home/data.json") is True
        # Tickers
        assert _is_file_path("AAPL") is False
        assert _is_file_path("MSFT") is False
        assert _is_file_path("BRK.B") is False  # dot but not file ext


class TestSchemaStandardization:
    """Test column name mapping and type casting."""

    def test_yahoo_column_names(self):
        from finasys.sources.schema import standardize_columns

        df = pl.DataFrame({"Date": ["x"], "Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [1000]})
        result = standardize_columns(df)
        assert "timestamp" in result.columns
        assert "open" in result.columns
        assert "close" in result.columns

    def test_adj_close_mapping(self):
        from finasys.sources.schema import standardize_columns

        df = pl.DataFrame({"Date": ["x"], "Adj Close": [100.0]})
        result = standardize_columns(df)
        assert "close" in result.columns

    def test_ticker_symbol_mapping(self):
        from finasys.sources.schema import standardize_columns

        df = pl.DataFrame({"Ticker": ["AAPL"], "Close": [100.0]})
        result = standardize_columns(df)
        assert "symbol" in result.columns

    def test_validate_ohlcv_pass(self):
        from finasys.sources.schema import validate_ohlcv

        df = pl.DataFrame({"timestamp": [date(2024, 1, 1)], "close": [100.0]})
        assert validate_ohlcv(df) is df

    def test_validate_ohlcv_fail(self):
        from finasys.sources.schema import validate_ohlcv

        with pytest.raises(ValueError, match="Missing"):
            validate_ohlcv(pl.DataFrame({"price": [100]}))

    def test_detect_ohlcv_true(self):
        from finasys.sources.schema import detect_ohlcv_schema

        df = pl.DataFrame({"Date": ["x"], "Close": [100.0]})
        assert detect_ohlcv_schema(df) is True

    def test_detect_ohlcv_false(self):
        from finasys.sources.schema import detect_ohlcv_schema

        df = pl.DataFrame({"name": ["Alice"], "age": [30]})
        assert detect_ohlcv_schema(df) is False

    def test_cast_string_dates(self):
        from finasys.sources.schema import cast_ohlcv_types

        df = pl.DataFrame({"timestamp": ["2024-01-01"], "close": ["100.5"]})
        result = cast_ohlcv_types(df)
        assert result["close"].dtype == pl.Float64


class TestCache:
    """Test DuckDB caching layer."""

    @pytest.fixture
    def cache_df(self):
        """A small clean DataFrame suitable for DuckDB cache (Date type timestamp)."""
        return pl.DataFrame(
            {
                "timestamp": pl.date_range(date(2024, 1, 1), date(2024, 4, 9), eager=True)[:100],
                "open": [100.0 + i * 0.1 for i in range(100)],
                "high": [101.0 + i * 0.1 for i in range(100)],
                "low": [99.0 + i * 0.1 for i in range(100)],
                "close": [100.5 + i * 0.1 for i in range(100)],
                "volume": [1e6] * 100,
                "symbol": ["CTEST"] * 100,
            }
        )

    def test_cache_put_and_get(self, cache_df):
        from finasys.sources.cache import cache_clear, cache_get, cache_put

        cache_clear("CTEST")
        cache_put(cache_df, "CTEST")
        result = cache_get("CTEST")
        assert result is not None
        assert result.height == 100
        cache_clear("CTEST")

    def test_cache_get_empty(self):
        from finasys.sources.cache import cache_clear, cache_get

        cache_clear("NONEXISTENT")
        result = cache_get("NONEXISTENT")
        assert result is None

    def test_cache_clear_specific(self, cache_df):
        from finasys.sources.cache import cache_clear, cache_get, cache_put

        cache_put(cache_df, "SYM_A")
        cache_clear("SYM_A")
        assert cache_get("SYM_A") is None

    def test_cache_date_range(self, cache_df):
        from finasys.sources.cache import cache_clear, cache_get, cache_put

        cache_clear("RANGETEST")
        cache_put(cache_df, "RANGETEST")
        result = cache_get("RANGETEST", start="2024-02-01", end="2024-03-01")
        assert result is not None
        assert result.height < cache_df.height
        cache_clear("RANGETEST")


# ============================================================
# 2. FEATURES MODULE -- INDICATORS (15 functions)
# ============================================================


class TestSMA:
    def test_default(self, sample_ohlcv):
        from finasys.features import sma

        r = sma(sample_ohlcv)
        assert "sma_20" in r.columns

    def test_custom_period(self, sample_ohlcv):
        from finasys.features import sma

        r = sma(sample_ohlcv, period=10)
        assert "sma_10" in r.columns

    def test_custom_column(self, sample_ohlcv):
        from finasys.features import sma

        r = sma(sample_ohlcv, column="open")
        assert "sma_20" in r.columns

    def test_warmup_nulls(self, sample_ohlcv):
        from finasys.features import sma

        r = sma(sample_ohlcv, period=10)
        assert r["sma_10"][:9].null_count() == 9
        assert r["sma_10"][9] is not None

    def test_correctness(self):
        from finasys.features import sma

        df = pl.DataFrame(
            {"timestamp": [date(2024, 1, i) for i in range(1, 6)], "close": [10.0, 20.0, 30.0, 40.0, 50.0]}
        )
        r = sma(df, period=3)
        assert abs(r["sma_3"][2] - 20.0) < 0.01  # (10+20+30)/3
        assert abs(r["sma_3"][4] - 40.0) < 0.01  # (30+40+50)/3

    def test_multi_symbol(self, multi_sym):
        from finasys.features import sma

        r = sma(multi_sym, period=5)
        # Each symbol should have its own SMA
        aaa = r.filter(pl.col("symbol") == "AAA")["sma_5"].drop_nulls()
        bbb = r.filter(pl.col("symbol") == "BBB")["sma_5"].drop_nulls()
        # BBB prices are ~200, AAA ~100, so SMAs should differ significantly
        assert abs(aaa.mean() - bbb.mean()) > 50


class TestEMA:
    def test_default(self, sample_ohlcv):
        from finasys.features import ema

        r = ema(sample_ohlcv)
        assert "ema_20" in r.columns
        assert r["ema_20"].drop_nulls().len() > 0

    def test_multi_symbol(self, multi_sym):
        from finasys.features import ema

        r = ema(multi_sym, period=10)
        aaa = r.filter(pl.col("symbol") == "AAA")["ema_10"].drop_nulls()
        bbb = r.filter(pl.col("symbol") == "BBB")["ema_10"].drop_nulls()
        assert abs(aaa.mean() - bbb.mean()) > 50


class TestRSI:
    def test_default(self, sample_ohlcv):
        from finasys.features import rsi

        r = rsi(sample_ohlcv)
        assert "rsi_14" in r.columns

    def test_range_0_to_100(self, sample_ohlcv):
        from finasys.features import rsi

        r = rsi(sample_ohlcv)
        vals = r["rsi_14"].drop_nulls()
        assert vals.min() >= 0
        assert vals.max() <= 100

    def test_custom_period(self, sample_ohlcv):
        from finasys.features import rsi

        r = rsi(sample_ohlcv, period=7)
        assert "rsi_7" in r.columns

    def test_multi_symbol(self, multi_sym):
        from finasys.features import rsi

        r = rsi(multi_sym, period=14)
        for sym in ["AAA", "BBB"]:
            vals = r.filter(pl.col("symbol") == sym)["rsi_14"].drop_nulls()
            assert vals.min() >= 0
            assert vals.max() <= 100


class TestMACD:
    def test_default(self, sample_ohlcv):
        from finasys.features import macd

        r = macd(sample_ohlcv)
        assert "macd_line" in r.columns
        assert "macd_signal" in r.columns
        assert "macd_hist" in r.columns

    def test_custom_params(self, sample_ohlcv):
        from finasys.features import macd

        r = macd(sample_ohlcv, fast=8, slow=21, signal=5)
        assert "macd_line" in r.columns

    def test_hist_equals_line_minus_signal(self, sample_ohlcv):
        from finasys.features import macd

        r = macd(sample_ohlcv)
        valid = r.drop_nulls(subset=["macd_line", "macd_signal", "macd_hist"])
        diff = (valid["macd_line"] - valid["macd_signal"] - valid["macd_hist"]).abs()
        assert diff.max() < 1e-10

    def test_multi_symbol(self, multi_sym):
        from finasys.features import macd

        r = macd(multi_sym)
        for sym in ["AAA", "BBB"]:
            vals = r.filter(pl.col("symbol") == sym)["macd_line"].drop_nulls()
            assert vals.len() > 0


class TestBollinger:
    def test_default(self, sample_ohlcv):
        from finasys.features import bollinger

        r = bollinger(sample_ohlcv)
        assert all(c in r.columns for c in ["bb_middle", "bb_upper", "bb_lower"])

    def test_upper_ge_lower(self, sample_ohlcv):
        from finasys.features import bollinger

        r = bollinger(sample_ohlcv).drop_nulls(subset=["bb_upper", "bb_lower"])
        assert (r["bb_upper"] >= r["bb_lower"]).all()

    def test_custom_std(self, sample_ohlcv):
        from finasys.features import bollinger

        r1 = bollinger(sample_ohlcv, std=1.0).drop_nulls(subset=["bb_upper"])
        r2 = bollinger(sample_ohlcv, std=3.0).drop_nulls(subset=["bb_upper"])
        # Wider std = wider bands
        width1 = (r1["bb_upper"] - r1["bb_lower"]).mean()
        width2 = (r2["bb_upper"] - r2["bb_lower"]).mean()
        assert width2 > width1


class TestATR:
    def test_default(self, sample_ohlcv):
        from finasys.features import atr

        r = atr(sample_ohlcv)
        assert "atr_14" in r.columns

    def test_always_positive(self, sample_ohlcv):
        from finasys.features import atr

        r = atr(sample_ohlcv)
        vals = r["atr_14"].drop_nulls()
        assert (vals > 0).all()

    def test_custom_period(self, sample_ohlcv):
        from finasys.features import atr

        r = atr(sample_ohlcv, period=7)
        assert "atr_7" in r.columns


class TestVWAP:
    def test_default(self, sample_ohlcv):
        from finasys.features import vwap

        r = vwap(sample_ohlcv)
        assert "vwap" in r.columns
        # VWAP should be a reasonable price (within the overall price range)
        assert r["vwap"].drop_nulls().len() == r.height
        assert r["vwap"].min() > 0


class TestOBV:
    def test_default(self, sample_ohlcv):
        from finasys.features import obv

        r = obv(sample_ohlcv)
        assert "obv" in r.columns
        assert r["obv"].drop_nulls().len() > 0


class TestStochastic:
    def test_default(self, sample_ohlcv):
        from finasys.features import stochastic

        r = stochastic(sample_ohlcv)
        assert "stoch_k" in r.columns
        assert "stoch_d" in r.columns

    def test_k_range(self, sample_ohlcv):
        from finasys.features import stochastic

        r = stochastic(sample_ohlcv)
        k = r["stoch_k"].drop_nulls()
        assert k.min() >= 0
        assert k.max() <= 100

    def test_custom_periods(self, sample_ohlcv):
        from finasys.features import stochastic

        r = stochastic(sample_ohlcv, k_period=21, d_period=5)
        assert "stoch_k" in r.columns


class TestADX:
    def test_default(self, sample_ohlcv):
        from finasys.features import adx

        r = adx(sample_ohlcv)
        assert "adx_14" in r.columns
        assert "plus_di" in r.columns
        assert "minus_di" in r.columns

    def test_adx_non_negative(self, sample_ohlcv):
        from finasys.features import adx

        r = adx(sample_ohlcv)
        vals = r["adx_14"].drop_nulls()
        assert (vals >= 0).all()


class TestCCI:
    def test_default(self, sample_ohlcv):
        from finasys.features import cci

        r = cci(sample_ohlcv)
        assert "cci_20" in r.columns

    def test_custom_period(self, sample_ohlcv):
        from finasys.features import cci

        r = cci(sample_ohlcv, period=14)
        assert "cci_14" in r.columns


class TestWilliamsR:
    def test_default(self, sample_ohlcv):
        from finasys.features import williams_r

        r = williams_r(sample_ohlcv)
        assert "williams_r_14" in r.columns

    def test_range(self, sample_ohlcv):
        from finasys.features import williams_r

        r = williams_r(sample_ohlcv)
        vals = r["williams_r_14"].drop_nulls()
        assert vals.min() >= -100
        assert vals.max() <= 0


class TestMFI:
    def test_default(self, sample_ohlcv):
        from finasys.features import mfi

        r = mfi(sample_ohlcv)
        assert "mfi_14" in r.columns


class TestROC:
    def test_default(self, sample_ohlcv):
        from finasys.features import roc

        r = roc(sample_ohlcv)
        assert "roc_10" in r.columns

    def test_correctness(self):
        from finasys.features import roc

        df = pl.DataFrame({"timestamp": [date(2024, 1, i) for i in range(1, 4)], "close": [100.0, 110.0, 120.0]})
        r = roc(df, period=1)
        # ROC = (120 - 110) / 110 * 100 = 9.09%
        assert abs(r["roc_1"][2] - 9.0909) < 0.01


class TestMomentum:
    def test_default(self, sample_ohlcv):
        from finasys.features import momentum

        r = momentum(sample_ohlcv)
        assert "momentum_10" in r.columns

    def test_correctness(self):
        from finasys.features import momentum

        df = pl.DataFrame({"timestamp": [date(2024, 1, i) for i in range(1, 4)], "close": [100.0, 110.0, 120.0]})
        r = momentum(df, period=1)
        assert abs(r["momentum_1"][2] - 10.0) < 0.01


# ============================================================
# 3. FEATURES MODULE -- RETURNS (4 functions)
# ============================================================


class TestReturns:
    def test_single_period(self, sample_ohlcv):
        from finasys.features import returns

        r = returns(sample_ohlcv, periods=1)
        assert "returns_1d" in r.columns

    def test_multiple_periods(self, sample_ohlcv):
        from finasys.features import returns

        r = returns(sample_ohlcv, periods=[1, 5, 21])
        assert all(f"returns_{p}d" in r.columns for p in [1, 5, 21])

    def test_first_value_null(self, sample_ohlcv):
        from finasys.features import returns

        r = returns(sample_ohlcv, periods=1)
        assert r["returns_1d"][0] is None

    def test_correctness(self):
        from finasys.features import returns

        df = pl.DataFrame({"timestamp": [date(2024, 1, 1), date(2024, 1, 2)], "close": [100.0, 105.0]})
        r = returns(df, periods=1)
        assert abs(r["returns_1d"][1] - 0.05) < 0.0001

    def test_multi_symbol_isolation(self, multi_sym):
        from finasys.features import returns

        r = returns(multi_sym, periods=1)
        # No return should be huge (symbol contamination would cause ~100% returns)
        vals = r["returns_1d"].drop_nulls()
        assert vals.abs().max() < 0.5  # 50% max for synthetic data


class TestLogReturns:
    def test_default(self, sample_ohlcv):
        from finasys.features import log_returns

        r = log_returns(sample_ohlcv)
        assert "log_returns_1d" in r.columns

    def test_multiple_periods(self, sample_ohlcv):
        from finasys.features import log_returns

        r = log_returns(sample_ohlcv, periods=[1, 5])
        assert "log_returns_1d" in r.columns
        assert "log_returns_5d" in r.columns

    def test_multi_symbol(self, multi_sym):
        from finasys.features import log_returns

        r = log_returns(multi_sym, periods=1)
        vals = r["log_returns_1d"].drop_nulls()
        assert vals.abs().max() < 0.5


class TestCumulativeReturns:
    def test_default(self, sample_ohlcv):
        from finasys.features import cumulative_returns

        r = cumulative_returns(sample_ohlcv)
        assert "cumulative_returns" in r.columns
        # First value should be 0
        assert abs(r["cumulative_returns"][0]) < 1e-10

    def test_multi_symbol(self, multi_sym):
        from finasys.features import cumulative_returns

        r = cumulative_returns(multi_sym)
        # Each symbol's cumulative return should start at 0
        for sym in ["AAA", "BBB"]:
            first = r.filter(pl.col("symbol") == sym)["cumulative_returns"][0]
            assert abs(first) < 1e-10


class TestDrawdown:
    def test_default(self, sample_ohlcv):
        from finasys.features import drawdown

        r = drawdown(sample_ohlcv)
        assert "drawdown" in r.columns
        assert "max_drawdown" in r.columns

    def test_always_non_positive(self, sample_ohlcv):
        from finasys.features import drawdown

        r = drawdown(sample_ohlcv)
        assert (r["drawdown"].drop_nulls() <= 0).all()
        assert (r["max_drawdown"].drop_nulls() <= 0).all()

    def test_max_dd_monotonically_decreasing(self, sample_ohlcv):
        from finasys.features import drawdown

        r = drawdown(sample_ohlcv)
        mdd = r["max_drawdown"].drop_nulls().to_list()
        for i in range(1, len(mdd)):
            assert mdd[i] <= mdd[i - 1]  # max drawdown can only get worse


# ============================================================
# 4. FEATURES MODULE -- ROLLING, LAGS, CALENDAR, CROSS
# ============================================================


class TestRollingStats:
    def test_default(self, sample_ohlcv):
        from finasys.features import rolling_stats

        r = rolling_stats(sample_ohlcv, windows=21)
        assert "rolling_mean_21" in r.columns
        assert "rolling_std_21" in r.columns

    def test_multiple_windows(self, sample_ohlcv):
        from finasys.features import rolling_stats

        r = rolling_stats(sample_ohlcv, windows=[5, 10, 21])
        for w in [5, 10, 21]:
            assert f"rolling_mean_{w}" in r.columns

    def test_all_stats(self, sample_ohlcv):
        from finasys.features import rolling_stats

        r = rolling_stats(sample_ohlcv, windows=10, stats=["mean", "std", "min", "max", "skew", "zscore"])
        for stat in ["mean", "std", "min", "max", "skew", "zscore"]:
            assert f"rolling_{stat}_10" in r.columns

    def test_unknown_stat_raises(self, sample_ohlcv):
        from finasys.features import rolling_stats

        with pytest.raises(ValueError, match="Unknown"):
            rolling_stats(sample_ohlcv, windows=10, stats=["invalid"])

    def test_multi_symbol(self, multi_sym):
        from finasys.features import rolling_stats

        r = rolling_stats(multi_sym, windows=5)
        aaa = r.filter(pl.col("symbol") == "AAA")["rolling_mean_5"].drop_nulls()
        bbb = r.filter(pl.col("symbol") == "BBB")["rolling_mean_5"].drop_nulls()
        assert abs(aaa.mean() - bbb.mean()) > 50


class TestLags:
    def test_single_lag(self, sample_ohlcv):
        from finasys.features import lags

        r = lags(sample_ohlcv, columns="close", lags=1)
        assert "close_lag_1" in r.columns
        assert r["close_lag_1"][0] is None
        assert r["close_lag_1"][1] == sample_ohlcv["close"][0]

    def test_multiple_columns_and_lags(self, sample_ohlcv):
        from finasys.features import lags

        r = lags(sample_ohlcv, columns=["close", "volume"], lags=[1, 3, 5])
        assert "close_lag_1" in r.columns
        assert "close_lag_5" in r.columns
        assert "volume_lag_3" in r.columns

    def test_rejects_negative_lag(self):
        from finasys.features import lags

        df = pl.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="positive"):
            lags(df, columns="close", lags=-1)

    def test_rejects_zero_lag(self):
        from finasys.features import lags

        df = pl.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="positive"):
            lags(df, columns="close", lags=0)

    def test_multi_symbol_isolation(self, multi_sym):
        from finasys.features import lags

        r = lags(multi_sym, columns="close", lags=1)
        # First row of each symbol should be null
        for sym in ["AAA", "BBB"]:
            sym_df = r.filter(pl.col("symbol") == sym)
            assert sym_df["close_lag_1"][0] is None


class TestValidateNoLookahead:
    def test_rsi_no_lookahead(self, sample_ohlcv):
        from finasys.features import rsi, validate_no_lookahead

        full = rsi(sample_ohlcv, period=14)
        partial = rsi(sample_ohlcv.head(50), period=14)
        assert validate_no_lookahead(full, partial, ["rsi_14"]) is True

    def test_sma_no_lookahead(self, sample_ohlcv):
        from finasys.features import sma, validate_no_lookahead

        full = sma(sample_ohlcv, period=10)
        partial = sma(sample_ohlcv.head(60), period=10)
        assert validate_no_lookahead(full, partial, ["sma_10"]) is True


class TestCalendarFeatures:
    def test_default(self, sample_ohlcv):
        from finasys.features import calendar_features

        r = calendar_features(sample_ohlcv)
        assert "day_of_week" in r.columns
        assert "month" in r.columns
        assert "quarter" in r.columns
        assert "week_of_year" in r.columns
        assert "is_month_start" in r.columns
        assert "is_month_end" in r.columns
        assert "is_quarter_end" in r.columns

    def test_day_of_week_range(self, sample_ohlcv):
        from finasys.features import calendar_features

        r = calendar_features(sample_ohlcv)
        dow = r["day_of_week"].drop_nulls()
        assert dow.min() >= 1
        assert dow.max() <= 7


class TestCrossSectional:
    def test_cross_rank(self, multi_sym):
        from finasys.features import cross_rank

        r = cross_rank(multi_sym, column="close")
        assert "close_rank" in r.columns
        # At each timestamp, ranks should be 1 and 2
        for ts in r["timestamp"].unique().to_list()[:5]:
            ranks = r.filter(pl.col("timestamp") == ts)["close_rank"].sort().to_list()
            assert ranks == [1, 2]

    def test_cross_percentile(self, multi_sym):
        from finasys.features import cross_percentile

        r = cross_percentile(multi_sym, column="close")
        assert "close_percentile" in r.columns
        vals = r["close_percentile"].drop_nulls()
        assert vals.min() > 0
        assert vals.max() <= 1.0

    def test_cross_zscore(self, multi_sym):
        from finasys.features import cross_zscore

        r = cross_zscore(multi_sym, column="close")
        assert "close_zscore" in r.columns


# ============================================================
# 5. FEATURES MODULE -- add_all() + FeatureSet
# ============================================================


class TestAddAll:
    def test_default(self, sample_ohlcv):
        from finasys.features import add_all

        r = add_all(sample_ohlcv)
        # Should have indicators + returns
        assert "rsi_14" in r.columns
        assert "macd_line" in r.columns
        assert "bb_upper" in r.columns
        assert "returns_1d" in r.columns

    def test_no_indicators(self, sample_ohlcv):
        from finasys.features import add_all

        r = add_all(sample_ohlcv, indicators=False)
        assert "rsi_14" not in r.columns

    def test_no_returns(self, sample_ohlcv):
        from finasys.features import add_all

        r = add_all(sample_ohlcv, returns_=False)
        assert "returns_1d" not in r.columns

    def test_with_lags(self, sample_ohlcv):
        from finasys.features import add_all

        r = add_all(sample_ohlcv, lags_=[1, 5, 10])
        assert "close_lag_1" in r.columns
        assert "close_lag_10" in r.columns

    def test_with_rolling(self, sample_ohlcv):
        from finasys.features import add_all

        r = add_all(sample_ohlcv, rolling_windows=[7, 14])
        assert "rolling_mean_7" in r.columns
        assert "rolling_std_14" in r.columns

    def test_with_calendar(self, sample_ohlcv):
        from finasys.features import add_all

        r = add_all(sample_ohlcv, calendar=True)
        assert "day_of_week" in r.columns

    def test_everything_on(self, sample_ohlcv):
        from finasys.features import add_all

        r = add_all(sample_ohlcv, indicators=True, returns_=True, lags_=[1, 5], rolling_windows=[5, 21], calendar=True)
        assert r.width > 30

    def test_minimal_df(self, minimal_close):
        """add_all on a DataFrame with only timestamp + close (no OHLCV)."""
        from finasys.features import add_all

        r = add_all(minimal_close, indicators=True, returns_=True)
        assert "rsi_14" in r.columns
        assert "returns_1d" in r.columns


class TestFeatureSet:
    def test_create_and_transform(self, sample_ohlcv):
        from finasys.features import MACD, RSI, FeatureSet

        fs = FeatureSet([RSI(), MACD()])
        r = fs.transform(sample_ohlcv)
        assert "rsi_14" in r.columns
        assert "macd_line" in r.columns

    def test_add_chaining(self, sample_ohlcv):
        from finasys.features import RSI, FeatureSet, Returns

        fs = FeatureSet()
        fs.add(RSI(period=7)).add(Returns(periods=[1, 5]))
        assert len(fs) == 2
        r = fs.transform(sample_ohlcv)
        assert "rsi_7" in r.columns
        assert "returns_5d" in r.columns

    def test_all_step_types(self, sample_ohlcv):
        """Test every registered FeatureStep class."""
        from finasys.features import (
            ATR,
            MACD,
            RSI,
            BollingerBands,
            Calendar,
            FeatureSet,
            Lags,
            LogReturns,
            Returns,
            RollingStats,
        )

        fs = FeatureSet(
            [
                RSI(period=14),
                MACD(fast=12, slow=26, signal=9),
                BollingerBands(period=20, std=2.0),
                ATR(period=14),
                Returns(periods=[1, 5]),
                LogReturns(periods=1),
                RollingStats(windows=[5, 21], stats=["mean", "std"]),
                Lags(columns=["close"], lags=[1, 3]),
                Calendar(),
            ]
        )
        r = fs.transform(sample_ohlcv)
        assert "rsi_14" in r.columns
        assert "macd_line" in r.columns
        assert "bb_upper" in r.columns
        assert "atr_14" in r.columns
        assert "returns_1d" in r.columns
        assert "log_returns_1d" in r.columns
        assert "rolling_mean_5" in r.columns
        assert "close_lag_1" in r.columns
        assert "day_of_week" in r.columns

    def test_save_and_load(self, tmp_path):
        from finasys.features import RSI, FeatureSet, Returns

        fs = FeatureSet([RSI(period=7), Returns(periods=[1, 21])])
        path = str(tmp_path / "pipeline.json")
        fs.save(path)

        loaded = FeatureSet.load(path)
        assert len(loaded) == 2
        assert loaded.steps[0].name == "RSI"
        assert loaded.steps[0].params["period"] == 7
        assert loaded.steps[1].name == "Returns"

    def test_roundtrip_produces_identical_output(self, sample_ohlcv, tmp_path):
        from finasys.features import MACD, RSI, FeatureSet, Returns

        fs = FeatureSet([RSI(), MACD(), Returns(periods=[1, 5, 21])])
        path = str(tmp_path / "pipe.json")

        r1 = fs.transform(sample_ohlcv)
        fs.save(path)
        r2 = FeatureSet.load(path).transform(sample_ohlcv)
        assert r1.equals(r2)

    def test_load_unknown_step_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"steps": [{"name": "Nonexistent", "params": {}}]}))
        from finasys.features import FeatureSet

        with pytest.raises(ValueError, match="Unknown"):
            FeatureSet.load(str(path))

    def test_repr(self):
        from finasys.features import RSI, FeatureSet

        fs = FeatureSet([RSI(period=14)])
        r = repr(fs)
        assert "RSI" in r
        assert "14" in r

    def test_empty(self, sample_ohlcv):
        from finasys.features import FeatureSet

        fs = FeatureSet()
        r = fs.transform(sample_ohlcv)
        assert r.equals(sample_ohlcv)


# ============================================================
# 6. AGENTS MODULE
# ============================================================


class TestSummarize:
    def test_basic(self, sample_ohlcv):
        from finasys.agents import summarize

        s = summarize(sample_ohlcv)
        assert isinstance(s, str)
        assert "TEST" in s  # symbol
        assert "$" in s  # price

    def test_with_indicators(self, sample_ohlcv):
        from finasys.agents import summarize
        from finasys.features import add_all

        df = add_all(sample_ohlcv)
        s = summarize(df)
        assert "RSI" in s or "Indicators" in s

    def test_max_tokens(self, sample_ohlcv):
        from finasys.agents import summarize

        s = summarize(sample_ohlcv, max_tokens=10)
        assert len(s) <= 44  # 10*4 + "..."

    def test_includes_volatility(self, sample_ohlcv):
        from finasys.agents import summarize

        s = summarize(sample_ohlcv)
        assert "Volatility" in s or "%" in s

    def test_includes_range(self, sample_ohlcv):
        from finasys.agents import summarize

        s = summarize(sample_ohlcv)
        assert "Range" in s

    def test_multi_symbol(self, multi_sym):
        from finasys.agents import summarize

        s = summarize(multi_sym)
        assert "AAA" in s or "BBB" in s

    def test_no_symbol_column(self, minimal_close):
        from finasys.agents import summarize

        s = summarize(minimal_close)
        assert "Unknown" in s  # default when no symbol


class TestTools:
    def test_returns_list(self):
        from finasys.agents import tools

        t = tools()
        assert isinstance(t, list)
        assert len(t) == 4

    def test_openai_format(self):
        from finasys.agents import tools

        for t in tools():
            assert t["type"] == "function"
            f = t["function"]
            assert "name" in f
            assert "description" in f
            assert "parameters" in f
            assert f["parameters"]["type"] == "object"

    def test_with_symbols(self):
        from finasys.agents import tools

        t = tools(symbols=["AAPL", "GOOGL"])
        descs = " ".join(x["function"]["description"] for x in t)
        assert "AAPL" in descs

    def test_tool_names(self):
        from finasys.agents import tools

        names = {t["function"]["name"] for t in tools()}
        assert names == {"lookup_price", "get_technical_indicators", "compare_symbols", "get_summary"}

    def test_each_tool_has_required_params(self):
        from finasys.agents import tools

        for t in tools():
            params = t["function"]["parameters"]
            assert "properties" in params
            # Each tool should have at least one required param
            if "required" in params:
                assert len(params["required"]) >= 1


class TestContext:
    def test_markdown_default(self, sample_ohlcv):
        from finasys.agents import context

        c = context(sample_ohlcv, "What is the current price?")
        assert isinstance(c, str)
        assert "|" in c  # markdown table

    def test_json_format(self, sample_ohlcv):
        from finasys.agents import context

        c = context(sample_ohlcv, "price", format="json")
        parsed = json.loads(c)
        assert isinstance(parsed, list)

    def test_text_format(self, sample_ohlcv):
        from finasys.agents import context

        c = context(sample_ohlcv, "price", format="text")
        assert isinstance(c, str)

    def test_momentum_query_selects_indicators(self, sample_ohlcv):
        from finasys.agents import context
        from finasys.features import add_all

        df = add_all(sample_ohlcv)
        c = context(df, "What is the momentum trend?")
        assert "rsi_14" in c or "macd" in c or "close" in c

    def test_volatility_query(self, sample_ohlcv):
        from finasys.agents import context
        from finasys.features import atr, rolling_stats

        df = atr(sample_ohlcv)
        df = rolling_stats(df, windows=10)
        c = context(df, "How volatile is it?")
        assert "atr" in c or "std" in c or "close" in c

    def test_volume_query(self, sample_ohlcv):
        from finasys.agents import context

        c = context(sample_ohlcv, "What is the trading volume?")
        assert "volume" in c

    def test_max_tokens(self, sample_ohlcv):
        from finasys.agents import context

        c = context(sample_ohlcv, "price", max_tokens=50)
        assert len(c) <= 250  # 50*4 + some slack


class TestSchema:
    def test_basic(self, sample_ohlcv):
        from finasys.agents import schema

        s = schema(sample_ohlcv)
        assert "100" in s  # 100 rows
        assert "columns" in s.lower()

    def test_includes_dtypes(self, sample_ohlcv):
        from finasys.agents import schema

        s = schema(sample_ohlcv)
        assert "Float64" in s or "f64" in s

    def test_includes_time_range(self, sample_ohlcv):
        from finasys.agents import schema

        s = schema(sample_ohlcv)
        assert "2024" in s

    def test_includes_symbols(self, multi_sym):
        from finasys.agents import schema

        s = schema(multi_sym)
        assert "AAA" in s
        assert "BBB" in s

    def test_reports_nulls(self):
        from finasys.agents import schema

        df = pl.DataFrame(
            {
                "timestamp": [date(2024, 1, 1), date(2024, 1, 2)],
                "close": [100.0, None],
            }
        )
        s = schema(df)
        assert "null" in s.lower() or "Null" in s


# ============================================================
# 7. CONFIG MODULE
# ============================================================


class TestConfig:
    def test_default_config(self):
        from finasys.utils.config import config

        assert config.cache_enabled is True
        assert config.default_backend == "polars"
        assert "finasys" in str(config.cache_dir)

    def test_ensure_cache_dir(self, tmp_path):
        from finasys.utils.config import FinaSysConfig

        cfg = FinaSysConfig(cache_dir=tmp_path / "test_cache")
        p = cfg.ensure_cache_dir()
        assert p.exists()


# ============================================================
# 8. TYPES MODULE
# ============================================================


class TestTypes:
    def test_ohlcv_columns(self):
        from finasys.utils.types import OHLCV_COLUMNS

        assert "timestamp" in OHLCV_COLUMNS
        assert "close" in OHLCV_COLUMNS
        assert len(OHLCV_COLUMNS) == 6

    def test_column_aliases(self):
        from finasys.utils.types import COLUMN_ALIASES

        assert COLUMN_ALIASES["Date"] == "timestamp"
        assert COLUMN_ALIASES["Adj Close"] == "close"
        assert COLUMN_ALIASES["Volume"] == "volume"


# ============================================================
# 9. VERSION
# ============================================================


class TestVersion:
    def test_version_exists(self):
        import finasys as fs

        assert hasattr(fs, "__version__")
        assert fs.__version__ == "0.1.3"


# ============================================================
# 10. INTEGRATION: FULL PIPELINE END-TO-END
# ============================================================


class TestEndToEnd:
    def test_single_symbol_full_pipeline(self, sample_ohlcv):
        """Simulate a real user's complete workflow."""
        import finasys as fs

        # Step 1: Load (from DataFrame, simulating file load)
        df = sample_ohlcv

        # Step 2: Add all features
        df = fs.features.add_all(df, lags_=[1, 5], rolling_windows=[5, 21], calendar=True)
        assert df.width > 30

        # Step 3: Get summary
        summary = fs.agents.summarize(df)
        assert len(summary) > 50

        # Step 4: Get tools
        tools = fs.agents.tools(symbols=["TEST"])
        assert len(tools) == 4

        # Step 5: Get context
        ctx = fs.agents.context(df, "What is the trend?")
        assert len(ctx) > 20

        # Step 6: Get schema
        sch = fs.agents.schema(df)
        assert "TEST" in sch

    def test_multi_symbol_full_pipeline(self, multi_sym):
        """Full pipeline on multi-symbol data."""
        import finasys as fs

        df = multi_sym

        # Features
        df = fs.features.rsi(df)
        df = fs.features.macd(df)
        df = fs.features.returns(df, periods=[1, 5])
        df = fs.features.lags(df, columns="close", lags=[1, 3])
        df = fs.features.cross_rank(df, column="close")
        df = fs.features.cross_zscore(df, column="returns_1d")

        # Verify no cross-symbol contamination
        for sym in ["AAA", "BBB"]:
            sym_df = df.filter(pl.col("symbol") == sym)
            rets = sym_df["returns_1d"].drop_nulls()
            assert rets.abs().max() < 0.5

        # Agents
        summary = fs.agents.summarize(df)
        assert isinstance(summary, str)

    def test_feature_set_pipeline(self, sample_ohlcv):
        """FeatureSet create -> transform -> save -> load -> re-transform."""
        import finasys as fs

        pipeline = fs.FeatureSet(
            [
                fs.features.RSI(period=14),
                fs.features.MACD(),
                fs.features.Returns(periods=[1, 5, 21]),
                fs.features.RollingStats(windows=[10], stats=["mean", "std"]),
                fs.features.Lags(columns=["close"], lags=[1, 2]),
            ]
        )

        r1 = pipeline.transform(sample_ohlcv)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            pipeline.save(path)
            pipeline2 = fs.FeatureSet.load(path)
            r2 = pipeline2.transform(sample_ohlcv)
            assert r1.equals(r2)
        finally:
            os.unlink(path)

    def test_csv_to_features_to_summary(self, sample_ohlcv, tmp_path):
        """Full file-based workflow: CSV -> load -> features -> summary."""
        import finasys as fs

        # Write CSV
        csv_path = tmp_path / "stock_data.csv"
        sample_ohlcv.drop("symbol").write_csv(csv_path)

        # Load
        df = fs.load(str(csv_path))
        assert isinstance(df, pl.DataFrame)
        assert "close" in df.columns

        # Features
        df = fs.features.rsi(df)
        df = fs.features.bollinger(df)
        df = fs.features.returns(df, periods=[1, 5])

        # Summary
        summary = fs.agents.summarize(df)
        assert "$" in summary

    def test_no_data_loss(self, sample_ohlcv):
        """Features should never drop rows."""
        import finasys as fs

        original_rows = sample_ohlcv.height

        df = sample_ohlcv
        df = fs.features.rsi(df)
        assert df.height == original_rows

        df = fs.features.macd(df)
        assert df.height == original_rows

        df = fs.features.bollinger(df)
        assert df.height == original_rows

        df = fs.features.returns(df, periods=[1, 5, 21])
        assert df.height == original_rows

        df = fs.features.lags(df, columns="close", lags=[1, 5])
        assert df.height == original_rows

        df = fs.features.add_all(df, calendar=True)
        assert df.height == original_rows
