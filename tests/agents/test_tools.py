"""Tests for agent tool definitions."""

from finasys.agents import tools


def test_tools_returns_list():
    result = tools()
    assert isinstance(result, list)
    assert len(result) > 0


def test_tools_openai_format():
    result = tools()
    for tool in result:
        assert "type" in tool
        assert tool["type"] == "function"
        assert "function" in tool
        func = tool["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func


def test_tools_with_symbols():
    result = tools(symbols=["AAPL", "GOOGL"])
    # Symbol names should appear in descriptions
    descriptions = " ".join(t["function"]["description"] for t in result)
    assert "AAPL" in descriptions


def test_tools_has_expected_functions():
    result = tools()
    names = {t["function"]["name"] for t in result}
    assert "lookup_price" in names
    assert "get_technical_indicators" in names
    assert "compare_symbols" in names
    assert "get_summary" in names
