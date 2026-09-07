#!/usr/bin/env python3
"""Run the optional Claude technology scout and save a markdown report."""
from __future__ import annotations

import asyncio
from pathlib import Path

from bist_hunter.agents.claude_runtime import run_claude_scout
from bist_hunter.agents.technology_scout import build_scout_prompt


async def main() -> None:
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    watchlist = (root / "config" / "research_watchlist.yaml").read_text(encoding="utf-8")
    prompt = build_scout_prompt(readme, watchlist)
    report = await run_claude_scout(prompt, root)
    output = root / "technology_scout_report.md"
    output.write_text(report + "\n", encoding="utf-8")
    print(f"Technology scout report written to {output}")


if __name__ == "__main__":
    asyncio.run(main())
