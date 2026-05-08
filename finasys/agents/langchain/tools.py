"""LangChain BaseTool implementations for finasys."""

from __future__ import annotations

from typing import Any


def create_tools(symbols: list[str] | None = None) -> list[Any]:
    """Create LangChain tools wrapping finasys functionality."""
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

    class AssessRiskInput(BaseModel):
        symbol: str = Field(description="Stock ticker symbol")
        start: str | None = Field(default=None, description="Start date YYYY-MM-DD")
        confidence: float = Field(default=0.95, description="VaR/CVaR confidence level")

    class PortfolioAnalysisInput(BaseModel):
        symbols: list[str] = Field(description="Stock ticker symbols")
        weights: dict[str, float] | None = Field(default=None, description="Optional symbol weight mapping")
        start: str | None = Field(default=None, description="Start date YYYY-MM-DD")

    class ScreenStocksInput(BaseModel):
        symbols: list[str] = Field(description="Stock ticker symbols")
        start: str | None = Field(default=None, description="Start date YYYY-MM-DD")
        min_sharpe: float | None = Field(default=None, description="Minimum Sharpe ratio")
        max_drawdown: float | None = Field(default=None, description="Maximum drawdown magnitude")
        rsi_min: float | None = Field(default=None, description="Minimum RSI")
        rsi_max: float | None = Field(default=None, description="Maximum RSI")

    class DataQualityInput(BaseModel):
        symbol: str = Field(description="Stock ticker symbol")
        start: str | None = Field(default=None, description="Start date YYYY-MM-DD")

    def _lookup_price(symbol: str, start: str | None = None, end: str | None = None) -> str:
        from finasys.agents.tools import execute_tool

        return execute_tool("lookup_price", {"symbol": symbol, "start": start, "end": end})

    def _get_indicators(symbol: str, indicators: list[str] | None = None, start: str | None = None) -> str:
        from finasys.agents.tools import execute_tool

        return execute_tool(
            "get_technical_indicators",
            {"symbol": symbol, "indicators": indicators or ["rsi", "macd", "bollinger"], "start": start},
        )

    def _get_summary(symbol: str, days: int = 252) -> str:
        from finasys.agents.tools import execute_tool

        return execute_tool("get_summary", {"symbol": symbol, "days": days})

    def _assess_risk(symbol: str, start: str | None = None, confidence: float = 0.95) -> str:
        from finasys.agents.tools import execute_tool

        return execute_tool("assess_risk", {"symbol": symbol, "start": start, "confidence": confidence})

    def _portfolio_analysis(
        symbols: list[str],
        weights: dict[str, float] | None = None,
        start: str | None = None,
    ) -> str:
        from finasys.agents.tools import execute_tool

        return execute_tool("portfolio_analysis", {"symbols": symbols, "weights": weights, "start": start})

    def _screen_stocks(
        symbols: list[str],
        start: str | None = None,
        min_sharpe: float | None = None,
        max_drawdown: float | None = None,
        rsi_min: float | None = None,
        rsi_max: float | None = None,
    ) -> str:
        from finasys.agents.tools import execute_tool

        return execute_tool(
            "screen_stocks",
            {
                "symbols": symbols,
                "start": start,
                "min_sharpe": min_sharpe,
                "max_drawdown": max_drawdown,
                "rsi_min": rsi_min,
                "rsi_max": rsi_max,
            },
        )

    def _data_quality_check(symbol: str, start: str | None = None) -> str:
        from finasys.agents.tools import execute_tool

        return execute_tool("data_quality_check", {"symbol": symbol, "start": start})

    def _profile_stock(symbol: str, start: str | None = None) -> str:
        from finasys.agents.tools import execute_tool

        return execute_tool("profile_stock", {"symbol": symbol, "start": start})

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
        StructuredTool.from_function(
            func=_assess_risk,
            name="assess_risk",
            description=f"Compute a stock risk report.{symbol_note}",
            args_schema=AssessRiskInput,
        ),
        StructuredTool.from_function(
            func=_portfolio_analysis,
            name="portfolio_analysis",
            description="Analyze portfolio return, volatility, and correlations.",
            args_schema=PortfolioAnalysisInput,
        ),
        StructuredTool.from_function(
            func=_screen_stocks,
            name="screen_stocks",
            description="Filter stocks by Sharpe, drawdown, and RSI criteria.",
            args_schema=ScreenStocksInput,
        ),
        StructuredTool.from_function(
            func=_data_quality_check,
            name="data_quality_check",
            description=f"Check financial data quality.{symbol_note}",
            args_schema=DataQualityInput,
        ),
        StructuredTool.from_function(
            func=_profile_stock,
            name="profile_stock",
            description=f"Generate a Smart Profiler summary.{symbol_note}",
            args_schema=DataQualityInput,
        ),
    ]
