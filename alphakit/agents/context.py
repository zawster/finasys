"""Context extraction for RAG-style LLM usage."""

from __future__ import annotations

import polars as pl


def context(
    df: pl.DataFrame,
    query: str,
    max_tokens: int = 500,
    format: str = "markdown",
) -> str:
    """Extract relevant context from a DataFrame based on a natural language query.

    Analyzes the query to determine what data subset and format would be
    most useful for an LLM to answer the question.

    Args:
        df: Financial DataFrame with alphakit standard columns.
        query: Natural language query (e.g., "What is the recent momentum?").
        max_tokens: Approximate token budget for the response.
        format: Output format -- "markdown", "json", or "text".

    Returns:
        Formatted string with relevant data context.
    """
    query_lower = query.lower()

    # Determine which data to extract based on query keywords
    if _matches_any(query_lower, ["recent", "latest", "current", "today", "now"]):
        subset = df.tail(10)
    elif _matches_any(query_lower, ["trend", "momentum", "direction"]):
        subset = df.tail(30)
    elif _matches_any(query_lower, ["volatil", "risk", "drawdown"]):
        subset = df.tail(60)
    elif _matches_any(query_lower, ["compar", "vs", "versus", "relative"]):
        subset = df.tail(21)
    elif _matches_any(query_lower, ["all", "full", "entire", "history"]):
        subset = df
    else:
        subset = df.tail(20)

    # Determine which columns are relevant
    relevant_cols = ["timestamp", "close"]

    if _matches_any(query_lower, ["volume", "liquidity", "traded"]):
        relevant_cols.append("volume")
    if _matches_any(query_lower, ["rsi", "overbought", "oversold"]):
        relevant_cols.extend([c for c in df.columns if "rsi" in c])
    if _matches_any(query_lower, ["macd", "crossover", "convergence"]):
        relevant_cols.extend([c for c in df.columns if "macd" in c])
    if _matches_any(query_lower, ["bollinger", "band", "squeeze"]):
        relevant_cols.extend([c for c in df.columns if "bb_" in c])
    if _matches_any(query_lower, ["return", "performance", "gain", "loss"]):
        relevant_cols.extend([c for c in df.columns if "return" in c])
    if _matches_any(query_lower, ["momentum", "trend", "direction"]):
        relevant_cols.extend([c for c in df.columns if c in (
            "rsi_14", "macd_line", "macd_signal", "sma_50", "momentum_10"
        )])
    if _matches_any(query_lower, ["volatil", "risk", "atr"]):
        relevant_cols.extend([c for c in df.columns if "atr" in c or "std" in c])
    if _matches_any(query_lower, ["drawdown"]):
        relevant_cols.extend([c for c in df.columns if "drawdown" in c])

    # Add symbol if present
    if "symbol" in df.columns:
        relevant_cols.insert(1, "symbol")

    # Deduplicate and filter to existing columns
    seen = set()
    unique_cols = []
    for c in relevant_cols:
        if c in df.columns and c not in seen:
            seen.add(c)
            unique_cols.append(c)

    # If no specific columns matched, include OHLCV
    if len(unique_cols) <= 2:
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns and c not in seen:
                seen.add(c)
                unique_cols.append(c)

    subset = subset.select([c for c in unique_cols if c in subset.columns])

    # Format output
    if format == "json":
        import json as _json
        # Convert to list of dicts using Polars (no pandas needed)
        rows = subset.to_dicts()
        # Convert date/datetime objects to strings for JSON serialization
        for row in rows:
            for k, v in row.items():
                if hasattr(v, "isoformat"):
                    row[k] = v.isoformat()
        return _json.dumps(rows, indent=2)
    elif format == "markdown":
        return _to_markdown(subset, max_tokens)
    else:
        return str(subset)


def _matches_any(text: str, keywords: list[str]) -> bool:
    """Check if any keyword appears in the text."""
    return any(kw in text for kw in keywords)


def _to_markdown(df: pl.DataFrame, max_tokens: int) -> str:
    """Convert a Polars DataFrame to a markdown table using Polars only."""
    # Build markdown table without pandas
    headers = df.columns
    md_lines = ["| " + " | ".join(headers) + " |"]
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in df.iter_rows():
        cells = []
        for v in row:
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        md_lines.append("| " + " | ".join(cells) + " |")
    md = "\n".join(md_lines)

    # Trim if over budget
    max_chars = max_tokens * 4
    if len(md) > max_chars:
        lines = md.split("\n")
        result = []
        total = 0
        for line in lines:
            if total + len(line) > max_chars:
                result.append("... (truncated)")
                break
            result.append(line)
            total += len(line)
        md = "\n".join(result)

    return md
