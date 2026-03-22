"""Generate LLM-ready summaries of financial DataFrames."""

from __future__ import annotations

import polars as pl


def summarize(df: pl.DataFrame, max_tokens: int | None = None) -> str:
    """Generate a structured text summary of a financial DataFrame.

    Produces a concise summary suitable for feeding into an LLM as context.
    Includes date range, key statistics, recent trends, and volatility.

    Args:
        df: DataFrame with standard alphakit columns (timestamp, close, etc.).
        max_tokens: Approximate token budget. If set, truncates the summary
                    to fit. Rough estimate: 1 token ~ 4 characters.

    Returns:
        A human-readable text summary.
    """
    parts: list[str] = []

    # Symbol
    symbol = "Unknown"
    if "symbol" in df.columns:
        symbols = df["symbol"].unique().to_list()
        if len(symbols) == 1:
            symbol = symbols[0]
        else:
            symbol = ", ".join(str(s) for s in symbols)

    # Date range
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        start = ts.min()
        end = ts.max()
        n_rows = df.height
        parts.append(f"{symbol} | {start} to {end} ({n_rows} trading days)")
    else:
        parts.append(f"{symbol} | {df.height} rows")

    # Current price and basic stats
    if "close" in df.columns:
        close = df["close"]
        current = close.item(-1)
        parts.append(f"Current price: ${current:.2f}")

        # High/Low range
        high = close.max()
        low = close.min()
        parts.append(f"Range: ${low:.2f} - ${high:.2f}")

        # Recent returns
        if df.height >= 2:
            prev = close.item(-2)
            daily_ret = (current - prev) / prev * 100
            parts.append(f"Last day: {daily_ret:+.2f}%")

        if df.height >= 6:
            week_ago = close.item(-5)
            weekly_ret = (current - week_ago) / week_ago * 100
            parts.append(f"Last 5 days: {weekly_ret:+.2f}%")

        if df.height >= 22:
            month_ago = close.item(-21)
            monthly_ret = (current - month_ago) / month_ago * 100
            parts.append(f"Last 21 days: {monthly_ret:+.2f}%")

        # Volatility (annualized from daily std of returns)
        if df.height >= 21:
            daily_returns = close.pct_change().drop_nulls()
            if daily_returns.len() > 0:
                vol = daily_returns.std() * (252**0.5) * 100
                parts.append(f"Volatility (annualized): {vol:.1f}%")

    # Technical indicators if present
    indicator_parts = []
    if "rsi_14" in df.columns:
        rsi_val = df["rsi_14"].drop_nulls().item(-1) if df["rsi_14"].drop_nulls().len() > 0 else None
        if rsi_val is not None:
            status = "overbought" if rsi_val > 70 else "oversold" if rsi_val < 30 else "neutral"
            indicator_parts.append(f"RSI(14): {rsi_val:.1f} ({status})")

    if "sma_50" in df.columns:
        sma_val = df["sma_50"].drop_nulls()
        if sma_val.len() > 0:
            sma50 = sma_val.item(-1)
            current_price = df["close"].item(-1)
            above_below = "above" if current_price > sma50 else "below"
            indicator_parts.append(f"Price is {above_below} 50-day SMA (${sma50:.2f})")

    if "macd_line" in df.columns and "macd_signal" in df.columns:
        macd_l = df["macd_line"].drop_nulls()
        macd_s = df["macd_signal"].drop_nulls()
        if macd_l.len() > 0 and macd_s.len() > 0:
            signal = "bullish" if macd_l.item(-1) > macd_s.item(-1) else "bearish"
            indicator_parts.append(f"MACD: {signal} crossover")

    if indicator_parts:
        parts.append("Indicators: " + " | ".join(indicator_parts))

    # Volume info
    if "volume" in df.columns:
        vol = df["volume"]
        avg_vol = vol.mean()
        last_vol = vol.item(-1)
        if avg_vol and avg_vol > 0:
            vol_ratio = last_vol / avg_vol
            parts.append(f"Volume: {last_vol:,.0f} ({vol_ratio:.1f}x avg)")

    summary = "\n".join(parts)

    # Token budget trimming
    if max_tokens is not None:
        max_chars = max_tokens * 4
        if len(summary) > max_chars:
            summary = summary[:max_chars] + "..."

    return summary
