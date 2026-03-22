"""alphakit.agents -- AI agent financial data tools.

Provides tool definitions, structured summaries, and context extraction
for AI agents (LangChain, OpenAI function calling, CrewAI, etc.).

Usage:
    import alphakit as ak

    df = ak.load("AAPL", start="2024-01-01")
    df = ak.features.add_all(df)

    # LLM-ready summary
    summary = ak.agents.summarize(df)

    # Tool definitions for OpenAI function calling
    tool_defs = ak.agents.tools(symbols=["AAPL", "GOOGL"])

    # Context extraction for RAG
    ctx = ak.agents.context(df, "What is the recent momentum?")

    # Schema for system prompts
    schema_desc = ak.agents.schema(df)
"""

from alphakit.agents.context import context
from alphakit.agents.schema import schema
from alphakit.agents.summarize import summarize
from alphakit.agents.tools import execute_tool, tools

__all__ = ["summarize", "tools", "execute_tool", "context", "schema"]
