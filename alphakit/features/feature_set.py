"""FeatureSet: composable, serializable feature pipeline."""

from __future__ import annotations

import json
from typing import Any

import polars as pl


class FeatureStep:
    """Base class for a single feature transformation step."""

    def __init__(self, name: str, **kwargs: Any):
        self.name = name
        self.params = kwargs

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"name": self.name, "params": self.params}

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params_str})"


class RSI(FeatureStep):
    def __init__(self, period: int = 14, column: str = "close"):
        super().__init__("RSI", period=period, column=column)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        from alphakit.features.indicators import rsi
        return rsi(df, **self.params)


class MACD(FeatureStep):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "close"):
        super().__init__("MACD", fast=fast, slow=slow, signal=signal, column=column)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        from alphakit.features.indicators import macd
        return macd(df, **self.params)


class BollingerBands(FeatureStep):
    def __init__(self, period: int = 20, std: float = 2.0, column: str = "close"):
        super().__init__("BollingerBands", period=period, std=std, column=column)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        from alphakit.features.indicators import bollinger
        return bollinger(df, **self.params)


class ATR(FeatureStep):
    def __init__(self, period: int = 14):
        super().__init__("ATR", period=period)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        from alphakit.features.indicators import atr
        return atr(df, **self.params)


class Returns(FeatureStep):
    def __init__(self, periods: list[int] | int = 1, column: str = "close"):
        super().__init__("Returns", periods=periods, column=column)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        from alphakit.features.returns import returns
        return returns(df, **self.params)


class LogReturns(FeatureStep):
    def __init__(self, periods: list[int] | int = 1, column: str = "close"):
        super().__init__("LogReturns", periods=periods, column=column)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        from alphakit.features.returns import log_returns
        return log_returns(df, **self.params)


class RollingStats(FeatureStep):
    def __init__(
        self,
        windows: list[int] | int = 21,
        column: str = "close",
        stats: list[str] | None = None,
    ):
        super().__init__("RollingStats", windows=windows, column=column, stats=stats)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        from alphakit.features.rolling import rolling_stats
        return rolling_stats(df, **self.params)


class Lags(FeatureStep):
    def __init__(self, columns: list[str] | str = "close", lags: list[int] | int = 1):
        super().__init__("Lags", columns=columns, lags=lags)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        from alphakit.features.lags import lags
        return lags(df, **self.params)


class Calendar(FeatureStep):
    def __init__(self, column: str = "timestamp"):
        super().__init__("Calendar", column=column)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        from alphakit.features.calendar import calendar_features
        return calendar_features(df, **self.params)


# Registry for deserialization
_STEP_REGISTRY: dict[str, type[FeatureStep]] = {
    "RSI": RSI,
    "MACD": MACD,
    "BollingerBands": BollingerBands,
    "ATR": ATR,
    "Returns": Returns,
    "LogReturns": LogReturns,
    "RollingStats": RollingStats,
    "Lags": Lags,
    "Calendar": Calendar,
}


class FeatureSet:
    """A composable, serializable pipeline of feature transformations.

    Usage:
        feature_set = ak.FeatureSet([
            ak.features.RSI(period=14),
            ak.features.MACD(),
            ak.features.Returns(periods=[1, 5, 21]),
        ])

        df = feature_set.transform(df)

        # Serialize for reproducibility
        feature_set.save("features.json")
        loaded = ak.FeatureSet.load("features.json")
    """

    def __init__(self, steps: list[FeatureStep] | None = None):
        self.steps: list[FeatureStep] = steps or []

    def add(self, step: FeatureStep) -> FeatureSet:
        """Add a step to the pipeline. Returns self for chaining."""
        self.steps.append(step)
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all feature steps sequentially."""
        for step in self.steps:
            df = step.transform(df)
        return df

    def save(self, path: str) -> None:
        """Serialize the feature set to a JSON file."""
        data = {"steps": [s.to_dict() for s in self.steps]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> FeatureSet:
        """Deserialize a feature set from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        steps = []
        for step_data in data["steps"]:
            name = step_data["name"]
            params = step_data["params"]
            if name not in _STEP_REGISTRY:
                raise ValueError(f"Unknown feature step: '{name}'")
            step_cls = _STEP_REGISTRY[name]
            steps.append(step_cls(**params))

        return cls(steps)

    def __repr__(self) -> str:
        steps_str = ", ".join(repr(s) for s in self.steps)
        return f"FeatureSet([{steps_str}])"

    def __len__(self) -> int:
        return len(self.steps)
