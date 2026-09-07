#!/usr/bin/env python3
"""Run the Bits100 independent AI research loop without touching production.

Flow: GPT-6 Astra + Claude independent scout -> quant review -> audit.
Model output remains research evidence only; deterministic backtests decide
whether a hypothesis survives promotion.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from bist_hunter.agents.ai_panel import build_panel_prompt, run_research_panel
from bist_hunter.agents.claude_runtime import run_claude_scout
from bist_hunter.agents.openai_runtime import run_openai_research
from bist_hunter.agents.quant_researcher import build_quant_research_prompt
from bist_hunter.agents.signal_auditor import build_auditor_prompt


async def _optional_claude(prompt: str, root: Path) -> str | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        return await run_claude_scout(prompt, root)
    except Exception as exc:  # pragma: no cover - integration boundary
        return f"CLAUDE_ERROR: {exc}"


async def _optional_gpt6(prompt: str) -> str | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        result = await asyncio.to_thread(
            run_openai_research,
            prompt,
            reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT", "high"),
        )
        return result.text
    except Exception as exc:  # pragma: no cover - integration boundary
        return f"GPT6_ERROR: {exc}"


async def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = (root / "README.md").read_text(encoding="utf-8")
    watchlist = (root / "config" / "research_watchlist.yaml").read_text(encoding="utf-8")

    scout_prompt = build_panel_prompt(
        market_snapshot=summary,
        candidate_symbols=watchlist,
        evidence="Repository architecture and configured research watchlist only.",
    )
    panel = await run_research_panel(scout_prompt, project_root=root)
    gpt6_scout = panel.gpt6.text if panel.gpt6 else f"GPT6_ERROR: {panel.errors}"
    claude_scout = panel.claude_text or "CLAUDE: unavailable"

    quant_prompt = build_quant_research_prompt(
        f"GPT-6 Astra independent scout:\n{gpt6_scout}\n\n"
        f"Claude independent scout:\n{claude_scout}",
        summary,
        watchlist,
    )
    gpt6_quant, claude_quant = await asyncio.gather(
        _optional_gpt6(quant_prompt),
        _optional_claude(quant_prompt, root),
    )
    quant = (
        f"## GPT-6 Astra\n{gpt6_quant or 'unavailable'}\n\n"
        f"## Claude\n{claude_quant or 'unavailable'}"
    )

    audit_prompt = build_auditor_prompt(quant)
    gpt6_audit, claude_audit = await asyncio.gather(
        _optional_gpt6(audit_prompt),
        _optional_claude(audit_prompt, root),
    )
    audit = (
        f"## GPT-6 Astra\n{gpt6_audit or 'unavailable'}\n\n"
        f"## Claude\n{claude_audit or 'unavailable'}"
    )

    output = root / "research_loop_report.md"
    output.write_text(
        "# Bits100 Research Loop\n\n"
        "## Independent Scout — GPT-6 Astra\n" + gpt6_scout +
        "\n\n## Independent Scout — Claude\n" + claude_scout +
        "\n\n## Quant Research Review\n" + quant +
        "\n\n## Signal Audit\n" + audit +
        "\n\n## Promotion Rule\n"
        "No model output is a trade signal. Candidate hypotheses must pass "
        "deterministic backtest, walk-forward OOS, risk gates and final holdout.\n",
        encoding="utf-8",
    )
    print(f"Research loop report written to {output}")


if __name__ == "__main__":
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        raise SystemExit("OPENAI_API_KEY or ANTHROPIC_API_KEY is required")
    asyncio.run(main())
