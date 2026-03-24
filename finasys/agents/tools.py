"""Tool definitions for AI agents (OpenAI function-calling format)."""

from __future__ import annotations

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

    return [
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

    else:
        return f"Unknown tool: {tool_name}"
