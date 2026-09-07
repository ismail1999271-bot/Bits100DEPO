import pytest

from bist_hunter.agents.ai_panel import build_panel_prompt


def test_panel_prompt_is_point_in_time_safe():
    prompt = build_panel_prompt(
        market_snapshot="2026-09-07 close snapshot",
        candidate_symbols="ASELS\nTHYAO",
        evidence="KAP filing",
    )
    assert "2026-09-07 close snapshot" in prompt
    assert "ASELS" in prompt
    assert "Never invent prices" in prompt
    assert "not a signal" in prompt


@pytest.mark.asyncio
async def test_panel_can_fail_closed(monkeypatch):
    from bist_hunter.agents import ai_panel

    def fail_gpt(*args, **kwargs):
        raise RuntimeError("missing")

    async def fail_claude(*args, **kwargs):
        raise RuntimeError("missing")

    monkeypatch.setattr(ai_panel, "run_openai_research", fail_gpt)
    monkeypatch.setattr(ai_panel, "run_claude_scout", fail_claude)
    result = await ai_panel.run_research_panel("test")
    assert result.gpt6 is None
    assert result.claude_text is None
    assert len(result.errors) == 2
