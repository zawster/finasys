"""Risk metrics for financial time series."""

from __future__ import annotations

import numpy as np
import polars as pl

from finasys.features.utils import symbol_aware
from finasys.stats._utils import (
    TRADING_DAYS,
    apply_per_symbol,
    kurtosis,
    norm_ppf,
    price_to_returns_np,
    skewness,
)


def sharpe_ratio(
    df: pl.DataFrame,
    window: int | None = None,
    risk_free_rate: float = 0.0,
    column: str = "close",
) -> pl.DataFrame | float:
    """Sharpe ratio: excess return per unit of risk.

    Args:
        df: DataFrame with price data.
        window: If None, returns scalar for entire series.
                If set, appends a rolling Sharpe column.
        risk_free_rate: Annual risk-free rate (e.g., 0.05 for 5%).
        column: Price column.

    Returns:
        Scalar float (window=None) or DataFrame with column appended.
    """
    daily_rf = risk_free_rate / TRADING_DAYS

    if window is None:
        col = df[column]
        rets = (col / col.shift(1) - 1).drop_nulls()
        if rets.is_empty():
            return 0.0
        excess = rets - daily_rf
        mean_excess = excess.mean()
        std = excess.std()
        if std is None or std < 1e-15:
            return 0.0
        return float(mean_excess / std * np.sqrt(TRADING_DAYS))

    col = pl.col(column)
    ret = col / col.shift(1) - 1 - daily_rf
    rolling_mean = ret.rolling_mean(window_size=window)
    rolling_std = ret.rolling_std(window_size=window)

    expr = (rolling_mean / rolling_std * np.sqrt(TRADING_DAYS)).alias(f"sharpe_{window}")
    return df.with_columns(symbol_aware(expr, df))


def sortino_ratio(
    df: pl.DataFrame,
    window: int | None = None,
    risk_free_rate: float = 0.0,
    column: str = "close",
) -> pl.DataFrame | float:
    """Sortino ratio: excess return per unit of downside risk.

    Unlike Sharpe, only penalizes downside volatility.

    Args:
        df: DataFrame with price data.
        window: If None, returns scalar. If set, appends rolling column.
        risk_free_rate: Annual risk-free rate.
        column: Price column.

    Returns:
        Scalar float or DataFrame.
    """
    daily_rf = risk_free_rate / TRADING_DAYS

    if window is None:
        col = df[column]
        rets = (col / col.shift(1) - 1).drop_nulls()
        if rets.is_empty():
            return 0.0
        excess = rets - daily_rf
        mean_excess = excess.mean()
        downside = excess.filter(excess < 0)
        if downside.is_empty() or downside.len() < 2:
            return float("inf") if mean_excess > 0 else 0.0
        dd_std = downside.std()
        if dd_std is None or dd_std < 1e-15:
            return float("inf") if mean_excess > 0 else 0.0
        return float(mean_excess / dd_std * np.sqrt(TRADING_DAYS))

    return apply_per_symbol(df, _rolling_sortino, window, daily_rf, column)


def _rolling_sortino(
    df: pl.DataFrame,
    window: int,
    daily_rf: float,
    column: str,
) -> pl.DataFrame:
    """Compute rolling Sortino for a single-symbol DataFrame."""
    rets = price_to_returns_np(df[column].to_numpy()) - daily_rf

    n = len(rets)
    result = np.full(n, np.nan)
    for i in range(window, n):
        chunk = rets[i - window + 1 : i + 1]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) < 2:
            continue
        mean_excess = valid.mean()
        downside = valid[valid < 0]
        if len(downside) < 2:
            result[i] = np.inf if mean_excess > 0 else 0.0
            continue
        dd_std = downside.std(ddof=1)
        if dd_std < 1e-15:
            result[i] = np.inf if mean_excess > 0 else 0.0
            continue
        result[i] = mean_excess / dd_std * np.sqrt(TRADING_DAYS)

    return df.with_columns(pl.Series(f"sortino_{window}", result))


def calmar_ratio(
    df: pl.DataFrame,
    column: str = "close",
) -> float:
    """Calmar ratio: annualized return / maximum drawdown.

    Args:
        df: DataFrame with price data.
        column: Price column.

    Returns:
        Calmar ratio as a float.
    """
    from finasys.features.returns import drawdown

    col = df[column]
    if col.len() < 2:
        return 0.0

    total_return = col.item(-1) / col.item(0) - 1
    n_days = col.len()
    ann_return = (1 + total_return) ** (TRADING_DAYS / n_days) - 1

    dd_df = drawdown(df, column=column)
    max_dd = abs(dd_df["max_drawdown"].min())

    if max_dd < 1e-15:
        return float("inf") if ann_return > 0 else 0.0

    return float(ann_return / max_dd)


def value_at_risk(
    df: pl.DataFrame,
    confidence: float = 0.95,
    method: str = "historical",
    window: int | None = None,
    column: str = "close",
) -> pl.DataFrame | float:
    """Value at Risk (VaR) at given confidence level.

    Methods:
        - historical: empirical quantile of returns
        - parametric: assumes normal distribution
        - cornish_fisher: adjusts for skewness and kurtosis

    Args:
        df: DataFrame with price data.
        confidence: Confidence level (e.g., 0.95 for 95%).
        method: VaR computation method.
        window: If None, returns scalar. If set, appends rolling column.
        column: Price column.

    Returns:
        Scalar float (negative, representing loss) or DataFrame.
    """
    _validate_var_method(method)
    alpha = 1 - confidence

    if window is None:
        col = df[column]
        rets = (col / col.shift(1) - 1).drop_nulls().to_numpy()
        if len(rets) < 2:
            return 0.0
        return float(_var_single(rets, alpha, method))

    return apply_per_symbol(df, _rolling_var, window, alpha, method, column)


def _validate_var_method(method: str) -> None:
    valid = ("historical", "parametric", "cornish_fisher")
    if method not in valid:
        raise ValueError(f"Unknown VaR method: '{method}'. Options: {', '.join(valid)}")


def _var_single(rets: np.ndarray, alpha: float, method: str) -> float:
    """Compute VaR for a single array of returns."""
    if method == "historical":
        return float(np.quantile(rets, alpha))
    elif method == "parametric":
        mu = rets.mean()
        sigma = rets.std()
        return float(mu + sigma * norm_ppf(alpha))
    else:  # cornish_fisher
        mu = rets.mean()
        sigma = rets.std()
        z = norm_ppf(alpha)
        s = skewness(rets)
        k = kurtosis(rets)
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
        return float(mu + sigma * z_cf)


def _rolling_var(
    df: pl.DataFrame,
    window: int,
    alpha: float,
    method: str,
    column: str,
) -> pl.DataFrame:
    """Compute rolling VaR for a single-symbol DataFrame."""
    rets = price_to_returns_np(df[column].to_numpy())
    n = len(rets)
    result = np.full(n, np.nan)
    for i in range(window, n):
        chunk = rets[i - window + 1 : i + 1]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) < 4:
            continue
        result[i] = _var_single(valid, alpha, method)

    return df.with_columns(pl.Series(f"var_{window}", result))


def cvar(
    df: pl.DataFrame,
    confidence: float = 0.95,
    window: int | None = None,
    column: str = "close",
) -> pl.DataFrame | float:
    """Conditional VaR (Expected Shortfall).

    The expected loss given that the loss exceeds VaR.

    Args:
        df: DataFrame with price data.
        confidence: Confidence level.
        window: If None, returns scalar. If set, appends rolling column.
        column: Price column.

    Returns:
        Scalar float or DataFrame.
    """
    alpha = 1 - confidence

    if window is None:
        col = df[column]
        rets = (col / col.shift(1) - 1).drop_nulls().to_numpy()
        if len(rets) < 2:
            return 0.0
        var_threshold = np.quantile(rets, alpha)
        tail = rets[rets <= var_threshold]
        return float(tail.mean()) if len(tail) > 0 else float(var_threshold)

    return apply_per_symbol(df, _rolling_cvar, window, alpha, column)


def _rolling_cvar(
    df: pl.DataFrame,
    window: int,
    alpha: float,
    column: str,
) -> pl.DataFrame:
    """Compute rolling CVaR for a single-symbol DataFrame."""
    rets = price_to_returns_np(df[column].to_numpy())
    n = len(rets)
    result = np.full(n, np.nan)
    for i in range(window, n):
        chunk = rets[i - window + 1 : i + 1]
        valid = chunk[~np.isnan(chunk)]
        if len(valid) < 4:
            continue
        var_threshold = np.quantile(valid, alpha)
        tail = valid[valid <= var_threshold]
        result[i] = tail.mean() if len(tail) > 0 else var_threshold

    return df.with_columns(pl.Series(f"cvar_{window}", result))


def max_drawdown_duration(
    df: pl.DataFrame,
    column: str = "close",
) -> pl.DataFrame:
    """Drawdown duration tracking.

    Appends columns:
        - dd_duration: bars in current drawdown (0 when at new high)
        - dd_max_duration: longest drawdown duration so far

    Args:
        df: DataFrame with price data.
        column: Price column.

    Returns:
        DataFrame with duration columns appended.
    """
    return apply_per_symbol(df, _dd_duration, column)


def _dd_duration(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """Compute drawdown duration for a single-symbol DataFrame."""
    prices = df[column].to_numpy()
    n = len(prices)

    durations = np.zeros(n, dtype=np.int32)
    max_durations = np.zeros(n, dtype=np.int32)

    running_max = prices[0]
    current_dd = 0
    max_dd_dur = 0

    for i in range(n):
        if prices[i] >= running_max:
            running_max = prices[i]
            current_dd = 0
        else:
            current_dd += 1

        durations[i] = current_dd
        max_dd_dur = max(max_dd_dur, current_dd)
        max_durations[i] = max_dd_dur

    return df.with_columns(
        pl.Series("dd_duration", durations),
        pl.Series("dd_max_duration", max_durations),
    )
