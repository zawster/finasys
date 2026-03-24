"""finasys -- From raw market data to ML-ready features in five lines of code.

A Polars-first Python toolkit for financial data processing,
feature engineering, and AI agent integration.

Quick start::

    import finasys as fs

    # Load stock data
    df = fs.load("AAPL", start="2024-01-01")

    # Add technical indicators
    df = fs.features.rsi(df, period=14)
    df = fs.features.macd(df)

    # Generate LLM-ready summary
    summary = fs.agents.summarize(df)
"""

from finasys import (
    agents,  # noqa: F401
    features,  # noqa: F401
)
from finasys.__metadata__ import __version__
from finasys.features.feature_set import FeatureSet
from finasys.sources import cache_clear, load

__all__ = [
    "__version__",
    "load",
    "cache_clear",
    "features",
    "agents",
    "FeatureSet",
]
