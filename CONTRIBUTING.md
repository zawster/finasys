# Contributing to finasys

Thank you for your interest in contributing to finasys! This guide will help you get started.

## Getting Started

### Finding Issues

- Look for issues labeled **"good first issue"** for beginner-friendly tasks
- Issues labeled **"help wanted"** are open for community contributions
- For larger changes, please open an issue first to discuss the approach

### Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/zawster/finasys.git
cd finasys
```

2. **Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. **Install in development mode**

```bash
pip install -e ".[dev]"
```

4. **Install pre-commit hooks**

```bash
pre-commit install
```

5. **Verify your setup**

```bash
pytest tests/ -v
```

## Making Changes

### Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 120 characters
- Follow existing code patterns and naming conventions
- All feature functions should be **symbol-aware** (use `.over("symbol")` for multi-symbol DataFrames)

### Writing Code

- **Indicators**: Add to `finasys/features/indicators.py`, use pure Polars expressions (no pandas/ta-lib)
- **Data sources**: Add to `finasys/sources/`, follow the `fs.load()` dispatcher pattern
- **Agent tools**: Add to `finasys/agents/`, ensure outputs are LLM-friendly
- All public functions need docstrings

### Writing Tests

- Tests mirror the source structure: `finasys/features/` -> `tests/features/`
- Use the fixtures in `tests/conftest.py` for synthetic OHLCV data
- No network calls in unit tests (mark network tests with `@pytest.mark.network`)
- Test both single-symbol and multi-symbol DataFrames

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/features/ -v

# With coverage
pytest tests/ --cov=finasys --cov-report=term-missing
```

## Pull Request Process

1. Create a new branch from `main`
2. Make your changes with clear, focused commits
3. Ensure all tests pass and pre-commit hooks are clean
4. Open a pull request with a clear description of what and why

### PR Title Convention

- `feat: add Keltner Channel indicator`
- `fix: correct RSI calculation for edge case`
- `docs: add tutorial for multi-symbol analysis`
- `test: add coverage for calendar features`
- `refactor: simplify rolling stats implementation`

## Adding a New Indicator

Here's a quick guide for the most common contribution -- adding a new technical indicator:

1. Add the function to `finasys/features/indicators.py`:

```python
def your_indicator(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Your Indicator Name.

    Brief description of what it measures.
    """
    expr = (
        # Your Polars expression here
    ).alias(f"your_indicator_{period}")
    return df.with_columns(symbol_aware(expr, df))
```

2. Export it in `finasys/features/__init__.py`
3. Add tests in `tests/features/test_indicators.py`
4. Add a `FeatureStep` class in `finasys/features/feature_set.py` (optional)

## Questions?

Open an issue or start a discussion. We're happy to help!
