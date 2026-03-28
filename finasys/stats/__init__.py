"""finasys.stats -- Risk and performance metrics for financial time series.

Provides standard risk-adjusted performance metrics used in
portfolio management, risk analysis, and quantitative research.

Usage:
    import finasys as fs

    df = fs.load("AAPL", start="2024-01-01")

    # Scalar metrics (whole-series)
    sharpe = fs.stats.sharpe_ratio(df)
    var = fs.stats.value_at_risk(df, confidence=0.95)

    # Rolling metrics (ML features)
    df = fs.stats.sharpe_ratio(df, window=63)
    df = fs.stats.value_at_risk(df, window=63)
"""

from finasys.stats.performance import alpha_beta, information_ratio
from finasys.stats.risk import (
    calmar_ratio,
    cvar,
    max_drawdown_duration,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)

__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "value_at_risk",
    "cvar",
    "max_drawdown_duration",
    "alpha_beta",
    "information_ratio",
]
