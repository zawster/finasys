"""finasys quickstart -- 5 lines from raw data to ML-ready features."""

import finasys as fs

# 1. Load stock data (cached locally with DuckDB)
df = fs.load("AAPL", start="2024-01-01")
print(f"Loaded {df.height} rows of AAPL data")
print(df.head())

# 2. Add all standard features (RSI, MACD, BB, returns, etc.)
df = fs.features.add_all(df)
print(f"\nAfter feature engineering: {df.width} columns")
print(df.columns)

# 3. Get an LLM-ready summary
summary = fs.agents.summarize(df)
print(f"\n--- LLM Summary ---\n{summary}")

# 4. Get OpenAI function-calling tool definitions
tools = fs.agents.tools(symbols=["AAPL"])
print(f"\n--- Agent Tools ---")
for t in tools:
    print(f"  - {t['function']['name']}: {t['function']['description'][:60]}...")

# 5. Get schema for system prompts
schema = fs.agents.schema(df)
print(f"\n--- Schema ---\n{schema}")
