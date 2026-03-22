"""alphakit -- From raw market data to ML-ready features in five lines of code.

A Polars-first Python toolkit for financial data processing,
feature engineering, and AI agent integration.

Quick start::

    import alphakit as ak

    # Load stock data
    df = ak.load("AAPL", start="2024-01-01")

    # Add technical indicators
    df = ak.features.rsi(df, period=14)
    df = ak.features.macd(df)

    # Generate LLM-ready summary
    summary = ak.agents.summarize(df)
"""

from alphakit import (
    agents,  # noqa: F401
    features,  # noqa: F401
)
from alphakit.__metadata__ import __version__
from alphakit.features.feature_set import FeatureSet
from alphakit.sources import cache_clear, load

__all__ = [
    "__version__",
    "load",
    "cache_clear",
    "features",
    "agents",
    "FeatureSet",
]
