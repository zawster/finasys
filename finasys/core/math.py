"""Shared math utilities for finasys."""

from __future__ import annotations

import numpy as np


def price_to_returns_np(prices: np.ndarray) -> np.ndarray:
    """Convert a price array to a returns array (first element is NaN)."""
    n = len(prices)
    rets = np.empty(n)
    rets[0] = np.nan
    rets[1:] = prices[1:] / prices[:-1] - 1
    return rets


def skewness(arr: np.ndarray) -> float:
    """Compute skewness of an array."""
    m = arr.mean()
    diffs = arr - m
    m2 = np.mean(diffs**2)
    if m2 < 1e-15:
        return 0.0
    return float(np.mean(diffs**3) / m2**1.5)


def kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis of an array."""
    m = arr.mean()
    diffs = arr - m
    m2 = np.mean(diffs**2)
    if m2 < 1e-15:
        return 0.0
    return float(np.mean(diffs**4) / m2**2 - 3.0)


def norm_ppf(alpha: float) -> float:
    """Standard normal inverse CDF (percent point function).

    Rational approximation (Abramowitz and Stegun 26.2.23).
    Avoids scipy dependency.
    """
    if alpha <= 0 or alpha >= 1:
        return 0.0
    if alpha > 0.5:
        return -norm_ppf(1 - alpha)

    t = np.sqrt(-2 * np.log(alpha))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return -(t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3))
