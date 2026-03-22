"""LangChain integration for alphakit agent tools.

Requires: pip install alphakit[langchain]

Usage:
    from alphakit.agents.langchain import get_tools
    tools = get_tools(symbols=["AAPL", "GOOGL"])
    # Returns List[BaseTool] ready for LangChain agents
"""

from __future__ import annotations

from typing import Any


def get_tools(symbols: list[str] | None = None) -> list[Any]:
    """Get alphakit tools as LangChain BaseTool instances.

    Args:
        symbols: Optional list of supported symbols.

    Returns:
        List of LangChain BaseTool instances.

    Raises:
        ImportError: If langchain-core is not installed.
    """
    try:
        from alphakit.agents.langchain.tools import create_tools
    except ImportError:
        raise ImportError(
            "LangChain integration requires langchain-core. Install it with: pip install alphakit[langchain]"
        )

    return create_tools(symbols=symbols)


__all__ = ["get_tools"]
