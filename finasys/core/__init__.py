"""finasys.core -- Core constants, types, configuration, and shared utilities."""

from finasys.core.config import FinaSysConfig, config
from finasys.core.constants import (
    COLUMN_ALIASES,
    OHLCV_COLUMNS,
    REQUIRED_COLUMNS,
    TRADING_DAYS,
)
from finasys.core.dataframe import apply_per_symbol, has_multi_symbols, symbol_aware
from finasys.core.math import kurtosis, norm_ppf, price_to_returns_np, skewness
from finasys.core.types import Backend, PolarsFrame, Ticker

__all__ = [
    # Constants
    "TRADING_DAYS",
    "OHLCV_COLUMNS",
    "REQUIRED_COLUMNS",
    "COLUMN_ALIASES",
    # Types
    "PolarsFrame",
    "Backend",
    "Ticker",
    # Config
    "FinaSysConfig",
    "config",
    # DataFrame utilities
    "symbol_aware",
    "has_multi_symbols",
    "apply_per_symbol",
    # Math utilities
    "skewness",
    "kurtosis",
    "norm_ppf",
    "price_to_returns_np",
]
