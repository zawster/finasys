"""Portfolio analytics for multi-symbol financial DataFrames."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl

from finasys.core.constants import TRADING_DAYS

__all__ = [
    "correlation_matrix",
    "covariance_matrix",
    "rolling_correlation",
    "portfolio_returns",
    "equal_weight_returns",
    "minimum_variance_weights",
]


def _validate_multi_symbol(df: pl.DataFrame, column: str) -> None:
    missing = [c for c in ["timestamp", "symbol", column] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for portfolio analytics: {missing}")
    if df["symbol"].n_unique() < 2:
        raise ValueError("Portfolio analytics require at least two symbols")


def _returns_wide(df: pl.DataFrame, column: str = "close") -> pl.DataFrame:
    """Return a wide timestamp x symbol table of simple returns."""
    _validate_multi_symbol(df, column)
    returns_df = (
        df.sort(["symbol", "timestamp"])
        .with_columns((pl.col(column) / pl.col(column).shift(1).over("symbol") - 1).alias("_return"))
        .select("timestamp", "symbol", "_return")
    )
    return returns_df.pivot(index="timestamp", on="symbol", values="_return", aggregate_function="first").sort(
        "timestamp"
    )


def _numeric_matrix(wide: pl.DataFrame) -> tuple[list[str], np.ndarray]:
    symbols = [c for c in wide.columns if c != "timestamp"]
    arr = wide.select(symbols).to_numpy()
    mask = ~np.isnan(arr).any(axis=1)
    return symbols, arr[mask]


def correlation_matrix(
    df: pl.DataFrame,
    column: str = "close",
    method: str = "pearson",
) -> pl.DataFrame:
    """Return a symbol-by-symbol return correlation matrix.

    Args:
        df: Multi-symbol DataFrame with timestamp, symbol, and price column.
        column: Price column used to compute returns.
        method: "pearson" or "spearman".
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")

    wide = _returns_wide(df, column)
    symbols, arr = _numeric_matrix(wide)
    if arr.shape[0] < 2:
        corr = np.full((len(symbols), len(symbols)), np.nan)
    else:
        if method == "spearman":
            arr = np.apply_along_axis(lambda x: pl.Series(x).rank("average").to_numpy(), 0, arr)
        corr = np.corrcoef(arr, rowvar=False)

    return pl.DataFrame({"symbol": symbols, **{sym: corr[:, i] for i, sym in enumerate(symbols)}})


def covariance_matrix(
    df: pl.DataFrame,
    column: str = "close",
    annualize: bool = True,
) -> pl.DataFrame:
    """Return a symbol-by-symbol return covariance matrix."""
    wide = _returns_wide(df, column)
    symbols, arr = _numeric_matrix(wide)
    if arr.shape[0] < 2:
        cov = np.full((len(symbols), len(symbols)), np.nan)
    else:
        cov = np.cov(arr, rowvar=False)
        if annualize:
            cov = cov * TRADING_DAYS
    return pl.DataFrame({"symbol": symbols, **{sym: cov[:, i] for i, sym in enumerate(symbols)}})


def rolling_correlation(
    df: pl.DataFrame,
    symbol_a: str,
    symbol_b: str,
    window: int = 63,
    column: str = "close",
) -> pl.DataFrame:
    """Return pairwise rolling return correlation for two symbols."""
    if window < 2:
        raise ValueError("window must be at least 2")

    wide = _returns_wide(df, column)
    a = symbol_a.upper()
    b = symbol_b.upper()
    if a not in wide.columns or b not in wide.columns:
        raise ValueError(f"Both symbols must exist in DataFrame: {a}, {b}")

    arr_a = wide[a].to_numpy()
    arr_b = wide[b].to_numpy()
    result = np.full(wide.height, np.nan)
    for i in range(window - 1, wide.height):
        x = arr_a[i - window + 1 : i + 1]
        y = arr_b[i - window + 1 : i + 1]
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() >= 2:
            result[i] = np.corrcoef(x[mask], y[mask])[0, 1]

    return wide.select("timestamp").with_columns(pl.Series(f"rolling_corr_{a}_{b}_{window}", result))


def portfolio_returns(
    df: pl.DataFrame,
    weights: Mapping[str, float],
    column: str = "close",
) -> pl.DataFrame:
    """Return weighted portfolio simple returns by timestamp."""
    if not weights:
        raise ValueError("weights must not be empty")

    wide = _returns_wide(df, column)
    normalized = {str(k).upper(): float(v) for k, v in weights.items()}
    missing = [sym for sym in normalized if sym not in wide.columns]
    if missing:
        raise ValueError(f"Weight symbols not found in DataFrame: {missing}")

    total_weight = sum(normalized.values())
    if abs(total_weight) < 1e-15:
        raise ValueError("weights must not sum to zero")
    normalized = {sym: weight / total_weight for sym, weight in normalized.items()}

    expr = sum(pl.col(sym).fill_null(0.0) * weight for sym, weight in normalized.items())
    return wide.with_columns(expr.alias("portfolio_returns")).select("timestamp", "portfolio_returns")


def equal_weight_returns(df: pl.DataFrame, column: str = "close") -> pl.DataFrame:
    """Return equal-weight portfolio simple returns by timestamp."""
    wide = _returns_wide(df, column)
    symbols = [c for c in wide.columns if c != "timestamp"]
    weights = {sym: 1.0 / len(symbols) for sym in symbols}
    return portfolio_returns(df, weights, column=column)


def minimum_variance_weights(df: pl.DataFrame, column: str = "close") -> dict[str, float]:
    """Return long-only unconstrained minimum-variance weights.

    Uses the pseudo-inverse for numerical stability when assets are highly correlated.
    """
    wide = _returns_wide(df, column)
    symbols, arr = _numeric_matrix(wide)
    if arr.shape[0] < 2:
        raise ValueError("At least two complete return observations are required")

    cov = np.cov(arr, rowvar=False) * TRADING_DAYS
    inv_cov = np.linalg.pinv(cov)
    ones = np.ones(len(symbols))
    denom = ones @ inv_cov @ ones
    if abs(denom) < 1e-15:
        raise ValueError("Unable to compute stable minimum-variance weights")
    weights = inv_cov @ ones / denom
    return {sym: float(weight) for sym, weight in zip(symbols, weights)}
