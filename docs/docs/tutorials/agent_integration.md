# AI Agent Integration

This tutorial shows how to use finasys with AI agents and LLMs.

## OpenAI Function Calling

```python
import finasys as fs

# Get tool definitions
tools = fs.agents.tools(symbols=["AAPL", "GOOGL", "MSFT"])

# Plug directly into OpenAI
# response = openai.chat.completions.create(
#     model="gpt-4",
#     messages=[{"role": "user", "content": "How is AAPL doing?"}],
#     tools=tools,
# )

# Execute tool calls
result = fs.agents.execute_tool("get_summary", {"symbol": "AAPL"})
print(result)
```

## LLM Context Generation

```python
df = fs.load("AAPL", start="2024-01-01")
df = fs.features.add_all(df)

# Summary for system prompts
summary = fs.agents.summarize(df)

# Targeted context extraction
momentum = fs.agents.context(df, "What is the recent momentum?")
risk = fs.agents.context(df, "How volatile has it been?")

# Schema for data-aware prompts
schema = fs.agents.schema(df)
```

## LangChain Integration

```python
from finasys.agents.langchain import get_tools

tools = get_tools(symbols=["AAPL", "GOOGL"])
# Returns List[BaseTool] ready for any LangChain agent
```
