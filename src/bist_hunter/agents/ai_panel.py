"""Independent AI research panel: GPT-6 Astra + Claude.

Both models produce research hypotheses independently. The numerical quant
engine remains the arbiter; this module never creates an executable order.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from .claude_runtime import run_claude_scout
from .openai_runtime import OpenAIResearchResult, run_openai_research


@dataclass(frozen=True, slots=True)
class ResearchPanelResult:
    gpt6: OpenAIResearchResult | None
    claude_text: str | None
    errors: tuple[str, ...] = ()


async def run_research_panel(
    prompt: str,
    *,
    project_root: str | Path = ".",
    gpt6_reasoning_effort: str = "high",
) -> ResearchPanelResult:
    """Run independent model reviews without granting execution authority."""
    gpt6_task = asyncio.to_thread(
        run_openai_research,
        prompt,
        reasoning_effort=gpt6_reasoning_effort,
    )
    claude_task = run_claude_scout(prompt, project_root=project_root)
    results = await asyncio.gather(gpt6_task, claude_task, return_exceptions=True)

    errors: list[str] = []
    gpt6: OpenAIResearchResult | None = None
    claude_text: str | None = None
    if isinstance(results[0], OpenAIResearchResult):
        gpt6 = results[0]
    else:
        errors.append(f"gpt6: {results[0]}")
    if isinstance(results[1], str):
        claude_text = results[1]
    else:
        errors.append(f"claude: {results[1]}")

    return ResearchPanelResult(gpt6=gpt6, claude_text=claude_text, errors=tuple(errors))


def build_panel_prompt(
    *,
    market_snapshot: str,
    candidate_symbols: str,
    evidence: str,
) -> str:
    """Build a common, point-in-time-safe prompt for both research models."""
    return f"""
BITS100 INDEPENDENT RESEARCH REVIEW

Objective:
Identify testable BIST research hypotheses from the supplied point-in-time data.
Do not provide an executable trade order.

MARKET SNAPSHOT:
{market_snapshot}

CANDIDATE SYMBOLS:
{candidate_symbols}

AVAILABLE EVIDENCE:
{evidence}

Return:
1. Top 3 hypotheses, each with mechanism and expected direction.
2. Evidence supporting and contradicting each hypothesis.
3. Required quantitative features and exact backtest design.
4. Leakage / survivorship / look-ahead risks.
5. What would falsify each hypothesis.
6. Confidence 0-100, explicitly separated from expected return.

Rules:
- Never invent prices, holdings, KAP facts, fund flows, or broker targets.
- If evidence is missing, say MISSING DATA.
- A hypothesis is not a signal until the deterministic quant engine validates it.
""".strip()
