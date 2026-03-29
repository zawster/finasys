"""Shared type aliases and protocols for finasys."""

from __future__ import annotations

from typing import Literal

import polars as pl

# The primary DataFrame type used throughout finasys
PolarsFrame = pl.DataFrame | pl.LazyFrame

# Backend selection for output format
Backend = Literal["polars", "pandas"]

# Ticker can be a single string or list of strings
Ticker = str | list[str]
