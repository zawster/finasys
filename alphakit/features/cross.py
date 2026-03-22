"""Cross-sectional features for multi-symbol datasets."""

from __future__ import annotations

import polars as pl


def cross_rank(
    df: pl.DataFrame, column: str = "close", method: str = "ordinal"
) -> pl.DataFrame:
    """Rank values across symbols at each timestamp.

    For multi-symbol DataFrames, ranks the given column within
    each timestamp. Useful for cross-sectional momentum strategies.

    Args:
        df: DataFrame with 'timestamp' and 'symbol' columns.
        column: Column to rank across symbols.
        method: Ranking method -- "ordinal", "min", "max", "dense", "average".
    """
    return df.with_columns(
        pl.col(column)
        .rank(method=method)
        .over("timestamp")
        .alias(f"{column}_rank")
    )


def cross_percentile(df: pl.DataFrame, column: str = "close") -> pl.DataFrame:
    """Percentile rank across symbols at each timestamp.

    Values range from 0 to 1, where 1 is the highest value.
    """
    return df.with_columns(
        (
            pl.col(column).rank(method="average").over("timestamp")
            / pl.col(column).count().over("timestamp")
        ).alias(f"{column}_percentile")
    )


def cross_zscore(df: pl.DataFrame, column: str = "close") -> pl.DataFrame:
    """Z-score across symbols at each timestamp.

    Standardizes values relative to the cross-sectional mean and std.
    """
    return df.with_columns(
        (
            (pl.col(column) - pl.col(column).mean().over("timestamp"))
            / pl.col(column).std().over("timestamp")
        ).alias(f"{column}_zscore")
    )
