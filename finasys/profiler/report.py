"""Data classes for financial data profiling reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ColumnProfile:
    """Statistical profile of a single column."""

    name: str
    dtype: str
    count: int
    null_count: int
    null_pct: float
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    quantiles: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualityReport:
    """Financial data quality assessment."""

    missing_dates: list[str] = field(default_factory=list)
    duplicate_rows: int = 0
    zero_volume_days: int = 0
    price_outliers: dict[str, int] = field(default_factory=dict)
    suspected_splits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DistributionReport:
    """Distribution analysis of returns."""

    returns_skewness: float = 0.0
    returns_kurtosis: float = 0.0
    jarque_bera_stat: float = 0.0
    jarque_bera_pvalue: float = 1.0
    is_normal: bool = True
    tail_ratio: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProfileReport:
    """Comprehensive financial data profile report."""

    shape: tuple[int, int]
    date_range: tuple[str, str]
    symbols: list[str]
    column_stats: dict[str, ColumnProfile]
    quality: DataQualityReport
    distribution: DistributionReport

    def to_dict(self) -> dict[str, Any]:
        result = {
            "shape": self.shape,
            "date_range": self.date_range,
            "symbols": self.symbols,
            "column_stats": {k: v.to_dict() for k, v in self.column_stats.items()},
            "quality": self.quality.to_dict(),
            "distribution": self.distribution.to_dict(),
        }
        return result
