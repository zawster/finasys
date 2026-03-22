# AI Agent Integration

This tutorial shows how to use alphakit with AI agents and LLMs.

## OpenAI Function Calling

```python
import alphakit as ak

# Get tool definitions
tools = ak.agents.tools(symbols=["AAPL", "GOOGL", "MSFT"])

# Plug directly into OpenAI
# response = openai.chat.completions.create(
#     model="gpt-4",
#     messages=[{"role": "user", "content": "How is AAPL doing?"}],
#     tools=tools,
# )

# Execute tool calls
result = ak.agents.execute_tool("get_summary", {"symbol": "AAPL"})
print(result)
```

## LLM Context Generation

```python
df = ak.load("AAPL", start="2024-01-01")
df = ak.features.add_all(df)

# Summary for system prompts
summary = ak.agents.summarize(df)

# Targeted context extraction
momentum = ak.agents.context(df, "What is the recent momentum?")
risk = ak.agents.context(df, "How volatile has it been?")

# Schema for data-aware prompts
schema = ak.agents.schema(df)
```

## LangChain Integration

```python
from alphakit.agents.langchain import get_tools

tools = get_tools(symbols=["AAPL", "GOOGL"])
# Returns List[BaseTool] ready for any LangChain agent
```
