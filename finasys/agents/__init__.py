"""finasys.agents -- AI agent financial data tools.

Provides tool definitions, structured summaries, and context extraction
for AI agents (LangChain, OpenAI function calling, CrewAI, etc.).

Usage:
    import finasys as fs

    df = fs.load("AAPL", start="2024-01-01")
    df = fs.features.add_all(df)

    # LLM-ready summary
    summary = fs.agents.summarize(df)

    # Tool definitions for OpenAI function calling
    tool_defs = fs.agents.tools(symbols=["AAPL", "GOOGL"])

    # Context extraction for RAG
    ctx = fs.agents.context(df, "What is the recent momentum?")

    # Schema for system prompts
    schema_desc = fs.agents.schema(df)
"""

from finasys.agents.context import context
from finasys.agents.schema import schema
from finasys.agents.summarize import summarize
from finasys.agents.tools import execute_tool, tools

__all__ = ["summarize", "tools", "execute_tool", "context", "schema"]
