"""Calendar and temporal features for financial time series."""

from __future__ import annotations

import polars as pl

from alphakit.features.utils import symbol_aware


def calendar_features(df: pl.DataFrame, column: str = "timestamp") -> pl.DataFrame:
    """Add calendar-based features from a date/datetime column.

    Appends: day_of_week, month, quarter, is_month_start, is_month_end,
    is_quarter_end, week_of_year.

    These features capture seasonal patterns in financial data
    (e.g., the "January effect", end-of-quarter rebalancing).
    """
    col = pl.col(column)

    # These are pure date extractions -- no shift needed, so no symbol_aware
    basic_exprs = [
        col.dt.weekday().alias("day_of_week"),
        col.dt.month().alias("month"),
        col.dt.quarter().alias("quarter"),
        col.dt.week().alias("week_of_year"),
        (col.dt.day() == 1).cast(pl.Int8).alias("is_month_start"),
    ]

    # These use shift, so need symbol_aware for multi-symbol
    month_end = (col.dt.month() != col.shift(-1).dt.month()).cast(pl.Int8).alias("is_month_end")
    quarter_end = (col.dt.quarter() != col.shift(-1).dt.quarter()).cast(pl.Int8).alias("is_quarter_end")

    return df.with_columns(
        *basic_exprs,
        symbol_aware(month_end, df),
        symbol_aware(quarter_end, df),
    )
