# Agents API

Tools and utilities for integrating financial data with AI agents and LLMs.

---

## `fs.agents.summarize(df, max_tokens=None)`

Generate an LLM-ready text summary of a financial DataFrame.

```python
df = fs.load("AAPL", start="2024-01-01")
df = fs.features.add_all(df)

summary = fs.agents.summarize(df)
```

**Example output:**
```
AAPL | 2024-01-02 to 2025-03-20 (302 trading days)
Current price: $247.99
Range: $163.51 - $285.92
Last day: -0.39%
Last 5 days: -0.85%
Last 21 days: -4.83%
Volatility (annualized): 24.1%
Indicators: RSI(14): 27.4 (oversold) | Price is below 50-day SMA ($261.03) | MACD: bearish crossover
Volume: 87,981,315 (0.7x avg)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | required | DataFrame with finasys standard columns |
| `max_tokens` | `int` | `None` | Approximate token budget (1 token ~ 4 chars) |

---

## `fs.agents.tools(symbols=None)`

Generate tool definitions in OpenAI function-calling format.

```python
tools = fs.agents.tools(symbols=["AAPL", "GOOGL", "MSFT"])

# Plug directly into OpenAI
# response = client.chat.completions.create(tools=tools, ...)
```

**Returns** a list of 4 tool definitions:

| Tool Name | Description |
|-----------|-------------|
| `lookup_price` | Get price data for a symbol |
| `get_technical_indicators` | Compute RSI, MACD, Bollinger, etc. |
| `compare_symbols` | Compare performance across symbols |
| `get_summary` | Get comprehensive analysis summary |

---

## `fs.agents.execute_tool(tool_name, arguments)`

Execute a tool call and return the result as a string.

```python
result = fs.agents.execute_tool("get_summary", {"symbol": "AAPL", "days": 60})
result = fs.agents.execute_tool("lookup_price", {"symbol": "GOOGL", "start": "2024-06-01"})
```

---

## `fs.agents.context(df, query, max_tokens=500, format="markdown")`

Extract relevant context from a DataFrame based on a natural language query. Intelligently selects columns and rows based on query keywords.

```python
# Momentum query -> selects RSI, MACD, SMA columns
ctx = fs.agents.context(df, "What is the recent momentum?")

# Volatility query -> selects ATR, rolling_std columns
ctx = fs.agents.context(df, "How volatile has it been?")

# Volume query -> selects volume column
ctx = fs.agents.context(df, "What is the trading volume?")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pl.DataFrame` | required | DataFrame with features |
| `query` | `str` | required | Natural language query |
| `max_tokens` | `int` | `500` | Token budget for output |
| `format` | `str` | `"markdown"` | Output format: `"markdown"`, `"json"`, or `"text"` |

---

## `fs.agents.schema(df)`

Generate a human-readable schema description for LLM system prompts.

```python
schema = fs.agents.schema(df)
```

**Example output:**
```
DataFrame with 302 rows and 37 columns.
Time range: 2024-01-02 to 2025-03-20.
Symbols: AAPL.
Columns:
  timestamp (Date)
  open (Float64)
  high (Float64)
  ...
  rsi_14 (Float64)
  macd_line (Float64)
```

---

## LangChain Integration

Requires: `pip install finasys[langchain]`

```python
from finasys.agents.langchain import get_tools

tools = get_tools(symbols=["AAPL", "GOOGL"])
# Returns List[BaseTool] ready for any LangChain agent
```
