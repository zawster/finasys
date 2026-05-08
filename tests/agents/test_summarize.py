"""Tests for agent summarization."""

from finasys.agents import summarize


def test_summarize_basic(ohlcv_df):
    result = summarize(ohlcv_df)
    assert isinstance(result, str)
    assert "AAPL" in result
    assert "$" in result


def test_summarize_with_indicators(ohlcv_df):
    from finasys.features import add_all

    df = add_all(ohlcv_df)
    result = summarize(df)

    assert "RSI" in result or "Indicators" in result


def test_summarize_max_tokens(ohlcv_df):
    result = summarize(ohlcv_df, max_tokens=20)
    # 20 tokens * 4 chars = 80 chars max
    assert len(result) <= 84  # 80 + "..."


def test_summarize_max_tokens_exact_budget(ohlcv_df):
    result = summarize(ohlcv_df, max_tokens=10)

    assert len(result) <= 44


def test_summarize_includes_returns(ohlcv_df):
    result = summarize(ohlcv_df)
    # Should include at least daily return info
    assert "%" in result


def test_summarize_include_profile(ohlcv_df):
    result = summarize(ohlcv_df, include_profile=True)

    assert "DATA PROFILE" in result


def test_summarize_include_profile_swallows_profile_errors(ohlcv_df, monkeypatch):
    import finasys.profiler

    def _raise(*args, **kwargs):
        raise RuntimeError("profile unavailable")

    monkeypatch.setattr(finasys.profiler, "profile_summary", _raise)

    result = summarize(ohlcv_df, include_profile=True)

    assert "DATA PROFILE" not in result
    assert "$" in result
