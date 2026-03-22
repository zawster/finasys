"""LangChain BaseTool implementations for alphakit."""

from __future__ import annotations

from typing import Any


def create_tools(symbols: list[str] | None = None) -> list[Any]:
    """Create LangChain tools wrapping alphakit functionality."""
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class LookupPriceInput(BaseModel):
        symbol: str = Field(description="Stock ticker symbol (e.g., 'AAPL')")
        start: str | None = Field(default=None, description="Start date YYYY-MM-DD")
        end: str | None = Field(default=None, description="End date YYYY-MM-DD")

    class GetIndicatorsInput(BaseModel):
        symbol: str = Field(description="Stock ticker symbol")
        indicators: list[str] = Field(
            default=["rsi", "macd", "bollinger"],
            description="Indicators to compute: rsi, macd, bollinger, sma, ema, atr",
        )
        start: str | None = Field(default=None, description="Start date YYYY-MM-DD")

    class GetSummaryInput(BaseModel):
        symbol: str = Field(description="Stock ticker symbol")
        days: int = Field(default=252, description="Trading days to analyze")

    def _lookup_price(symbol: str, start: str | None = None, end: str | None = None) -> str:
        from alphakit.agents.tools import execute_tool
        return execute_tool("lookup_price", {"symbol": symbol, "start": start, "end": end})

    def _get_indicators(
        symbol: str, indicators: list[str] | None = None, start: str | None = None
    ) -> str:
        from alphakit.agents.tools import execute_tool
        return execute_tool(
            "get_technical_indicators",
            {"symbol": symbol, "indicators": indicators or ["rsi", "macd", "bollinger"], "start": start},
        )

    def _get_summary(symbol: str, days: int = 252) -> str:
        from alphakit.agents.tools import execute_tool
        return execute_tool("get_summary", {"symbol": symbol, "days": days})

    symbol_note = ""
    if symbols:
        symbol_note = f" Available: {', '.join(symbols)}."

    return [
        StructuredTool.from_function(
            func=_lookup_price,
            name="lookup_price",
            description=f"Get stock price data.{symbol_note}",
            args_schema=LookupPriceInput,
        ),
        StructuredTool.from_function(
            func=_get_indicators,
            name="get_technical_indicators",
            description=f"Compute technical indicators for a stock.{symbol_note}",
            args_schema=GetIndicatorsInput,
        ),
        StructuredTool.from_function(
            func=_get_summary,
            name="get_summary",
            description=f"Get comprehensive stock analysis summary.{symbol_note}",
            args_schema=GetSummaryInput,
        ),
    ]
