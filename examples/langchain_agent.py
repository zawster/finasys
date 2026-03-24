"""Example: Using finasys tools with an OpenAI-compatible LLM.

This example shows how to use finasys's agent tools with OpenAI's
function calling API. No LangChain required for this basic usage.

For LangChain integration, see:
    from finasys.agents.langchain import get_tools
    tools = get_tools(symbols=["AAPL", "GOOGL"])
"""

import json

import finasys as fs


def demo_tool_execution():
    """Demonstrate how agent tools work end-to-end."""

    # 1. Get tool definitions (OpenAI function-calling format)
    tool_defs = fs.agents.tools(symbols=["AAPL", "GOOGL", "MSFT"])

    print("Available tools for the LLM:")
    for t in tool_defs:
        print(f"  - {t['function']['name']}")
    print()

    # 2. Simulate an LLM tool call
    # In production, the LLM would choose which tool to call.
    # Here we'll execute each one manually.

    print("--- lookup_price ---")
    result = fs.agents.execute_tool("lookup_price", {"symbol": "AAPL", "start": "2024-06-01"})
    print(result[:500])
    print()

    print("--- get_summary ---")
    result = fs.agents.execute_tool("get_summary", {"symbol": "AAPL", "days": 60})
    print(result)
    print()

    print("--- get_technical_indicators ---")
    result = fs.agents.execute_tool(
        "get_technical_indicators",
        {"symbol": "GOOGL", "indicators": ["rsi", "macd"]},
    )
    print(result[:500])


if __name__ == "__main__":
    demo_tool_execution()
