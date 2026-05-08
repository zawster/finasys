"""Tool definitions for AI agents (OpenAI function-calling format)."""

from __future__ import annotations

import json
from typing import Any

import polars as pl


def tools(symbols: list[str] | None = None) -> list[dict[str, Any]]:
    """Generate tool definitions in OpenAI function-calling format.

    These tools can be plugged directly into:
        openai.chat.completions.create(tools=tools)

    Args:
        symbols: Optional list of supported symbols to include in descriptions.

    Returns:
        List of tool definition dicts in OpenAI format.
    """
    symbol_desc = ""
    if symbols:
        symbol_desc = f" Available symbols: {', '.join(symbols)}."

    base_tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_price",
                "description": (f"Get current and historical price data for a stock symbol.{symbol_desc}"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., 'AAPL')",
                        },
                        "start": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format",
                        },
                        "end": {
                            "type": "string",
                            "description": "End date in YYYY-MM-DD format (optional, defaults to today)",
                        },
                    },
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_technical_indicators",
                "description": (
                    "Calculate technical indicators (RSI, MACD, Bollinger Bands, etc.) "
                    f"for a stock symbol.{symbol_desc}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol",
                        },
                        "indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "List of indicators to compute. Options: "
                                "rsi, macd, bollinger, sma, ema, atr, vwap, obv, stochastic"
                            ),
                        },
                        "start": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format",
                        },
                    },
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compare_symbols",
                "description": ("Compare price performance and key metrics across multiple stock symbols."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of stock ticker symbols to compare",
                        },
                        "start": {
                            "type": "string",
                            "description": "Start date in YYYY-MM-DD format",
                        },
                        "metric": {
                            "type": "string",
                            "enum": ["returns", "volatility", "sharpe", "drawdown"],
                            "description": "Comparison metric (default: returns)",
                        },
                    },
                    "required": ["symbols"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_summary",
                "description": (
                    "Get a comprehensive text summary of a stock's recent performance, "
                    "including price, returns, volatility, and technical indicators."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol",
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of trading days to analyze (default: 252)",
                        },
                    },
                    "required": ["symbol"],
                },
            },
        },
    ]

    phase3_tools = [
        {
            "type": "function",
            "function": {
                "name": "assess_risk",
                "description": f"Compute a risk report with Sharpe, Sortino, VaR, CVaR, and drawdown.{symbol_desc}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"},
                        "start": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                        "confidence": {
                            "type": "number",
                            "description": "VaR/CVaR confidence level, default 0.95",
                        },
                    },
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "portfolio_analysis",
                "description": "Analyze a portfolio's returns, volatility, and cross-symbol correlations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbols": {"type": "array", "items": {"type": "string"}},
                        "weights": {
                            "type": "object",
                            "description": "Optional mapping of symbol to portfolio weight",
                        },
                        "start": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                    },
                    "required": ["symbols"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "screen_stocks",
                "description": "Filter symbols by risk and momentum criteria such as Sharpe, drawdown, and RSI.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbols": {"type": "array", "items": {"type": "string"}},
                        "start": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                        "min_sharpe": {"type": "number", "description": "Minimum Sharpe ratio"},
                        "max_drawdown": {"type": "number", "description": "Maximum allowed drawdown magnitude"},
                        "rsi_min": {"type": "number", "description": "Minimum RSI value"},
                        "rsi_max": {"type": "number", "description": "Maximum RSI value"},
                    },
                    "required": ["symbols"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "data_quality_check",
                "description": f"Check financial data quality: gaps, outliers, splits, nulls, and duplicates.{symbol_desc}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"},
                        "start": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                    },
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "profile_stock",
                "description": f"Generate a full Smart Profiler text summary for a stock symbol.{symbol_desc}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"},
                        "start": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                    },
                    "required": ["symbol"],
                },
            },
        },
    ]

    return base_tools + phase3_tools


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool call and return the result as a string.

    This is a convenience function for handling tool calls from an LLM.
    It dispatches to the appropriate finasys function based on tool_name.

    Args:
        tool_name: The function name from the tool call.
        arguments: The parsed arguments dict.

    Returns:
        String result suitable for returning to the LLM.
    """
    import finasys as fs

    if tool_name == "lookup_price":
        symbol = arguments["symbol"]
        start = arguments.get("start")
        end = arguments.get("end")
        df = fs.load(symbol, start=start, end=end)
        # Return last 10 rows as a readable string
        recent = df.tail(10)
        return recent.to_pandas().to_string(index=False)

    elif tool_name == "get_technical_indicators":
        symbol = arguments["symbol"]
        start = arguments.get("start")
        indicators = arguments.get("indicators", ["rsi", "macd", "bollinger"])
        df = fs.load(symbol, start=start)

        for ind in indicators:
            func = getattr(fs.features, ind, None)
            if func is not None:
                df = func(df)

        recent = df.tail(5)
        return recent.to_pandas().to_string(index=False)

    elif tool_name == "compare_symbols":
        symbols = arguments["symbols"]
        start = arguments.get("start")
        df = fs.load(symbols, start=start)
        df = fs.features.returns(df, periods=[1, 5, 21])

        # Summary per symbol
        summary_parts = []
        for sym in symbols:
            sym_df = df.filter(pl.col("symbol") == sym.upper())
            if sym_df.is_empty():
                continue
            current = sym_df["close"].item(-1)
            ret_col = "returns_21d"
            ret_21 = sym_df[ret_col].item(-1) if ret_col in sym_df.columns else None
            line = f"{sym}: ${current:.2f}"
            if ret_21 is not None:
                line += f" (21d return: {ret_21 * 100:+.1f}%)"
            summary_parts.append(line)

        return "\n".join(summary_parts) if summary_parts else "No data available"

    elif tool_name == "get_summary":
        symbol = arguments["symbol"]
        days = arguments.get("days", 252)
        df = fs.load(symbol)
        df = df.tail(days)
        df = fs.features.add_all(df)
        return fs.agents.summarize(df)

    elif tool_name == "assess_risk":
        symbol = arguments["symbol"]
        start = arguments.get("start")
        confidence = arguments.get("confidence", 0.95)
        df = fs.load(symbol, start=start)
        dd_df = fs.features.drawdown(df)
        max_dd = dd_df["max_drawdown"].min() if "max_drawdown" in dd_df.columns else 0.0
        return "\n".join(
            [
                f"Risk report for {symbol.upper()}",
                f"Sharpe: {fs.stats.sharpe_ratio(df):.3f}",
                f"Sortino: {fs.stats.sortino_ratio(df):.3f}",
                f"Calmar: {fs.stats.calmar_ratio(df):.3f}",
                f"VaR ({confidence:.0%}): {fs.stats.value_at_risk(df, confidence=confidence):.4f}",
                f"CVaR ({confidence:.0%}): {fs.stats.cvar(df, confidence=confidence):.4f}",
                f"Max drawdown: {max_dd:.2%}",
            ]
        )

    elif tool_name == "portfolio_analysis":
        symbols = arguments["symbols"]
        start = arguments.get("start")
        weights = arguments.get("weights")
        df = fs.load(symbols, start=start)
        if weights:
            portfolio = fs.portfolio.portfolio_returns(df, weights)
        else:
            portfolio = fs.portfolio.equal_weight_returns(df)
        corr = fs.portfolio.correlation_matrix(df)
        returns_col = portfolio["portfolio_returns"].drop_nulls()
        total_return = (1 + returns_col).product() - 1 if returns_col.len() else 0.0
        volatility = returns_col.std() * (252**0.5) if returns_col.len() > 1 else 0.0
        return "\n".join(
            [
                f"Portfolio: {', '.join(symbols)}",
                f"Total return: {total_return:.2%}",
                f"Annualized volatility: {volatility:.2%}",
                "Correlation matrix:",
                str(corr),
            ]
        )

    elif tool_name == "screen_stocks":
        symbols = arguments["symbols"]
        start = arguments.get("start")
        min_sharpe = arguments.get("min_sharpe")
        max_drawdown = arguments.get("max_drawdown")
        rsi_min = arguments.get("rsi_min")
        rsi_max = arguments.get("rsi_max")
        rows = []
        for symbol in symbols:
            df = fs.load(symbol, start=start)
            sharpe = fs.stats.sharpe_ratio(df)
            dd_df = fs.features.drawdown(df)
            max_dd = abs(dd_df["max_drawdown"].min()) if "max_drawdown" in dd_df.columns else 0.0
            rsi_df = fs.features.rsi(df)
            rsi_vals = rsi_df["rsi_14"].drop_nulls()
            rsi_val = rsi_vals.item(-1) if rsi_vals.len() else None

            if min_sharpe is not None and sharpe < min_sharpe:
                continue
            if max_drawdown is not None and max_dd > max_drawdown:
                continue
            if rsi_val is not None and rsi_min is not None and rsi_val < rsi_min:
                continue
            if rsi_val is not None and rsi_max is not None and rsi_val > rsi_max:
                continue
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "sharpe": round(float(sharpe), 3),
                    "max_drawdown": round(float(max_dd), 4),
                    "rsi_14": round(float(rsi_val), 2) if rsi_val is not None else None,
                }
            )
        return json.dumps(rows, indent=2)

    elif tool_name == "data_quality_check":
        symbol = arguments["symbol"]
        start = arguments.get("start")
        df = fs.load(symbol, start=start)
        report = fs.quality.completeness_report(df)
        return json.dumps(report, indent=2, default=str)

    elif tool_name == "profile_stock":
        symbol = arguments["symbol"]
        start = arguments.get("start")
        df = fs.load(symbol, start=start)
        return fs.profiler.profile_summary(df)

    else:
        return f"Unknown tool: {tool_name}"
