#!/usr/bin/env python3
"""Run Scout -> Quant Researcher -> Signal Auditor without touching production."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from bist_hunter.agents.claude_runtime import run_claude_scout
from bist_hunter.agents.quant_researcher import build_quant_research_prompt
from bist_hunter.agents.signal_auditor import build_auditor_prompt
from bist_hunter.agents.technology_scout import build_scout_prompt


async def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = (root / "README.md").read_text(encoding="utf-8")
    watchlist = (root / "config" / "research_watchlist.yaml").read_text(encoding="utf-8")
    scout = await run_claude_scout(build_scout_prompt(summary, watchlist), root)
    quant_prompt = build_quant_research_prompt(scout, summary, watchlist)
    quant = await run_claude_scout(quant_prompt, root)
    audit = await run_claude_scout(build_auditor_prompt(quant), root)
    output = root / "research_loop_report.md"
    output.write_text(
        "# Bits100 Research Loop\n\n"
        "## Technology Scout\n" + scout +
        "\n\n## Quant Researcher\n" + quant +
        "\n\n## Signal Auditor\n" + audit + "\n",
        encoding="utf-8",
    )
    print(f"Research loop report written to {output}")


if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required for the automated research loop")
    asyncio.run(main())
