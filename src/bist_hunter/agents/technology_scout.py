"""Technology-scout agent contract.

The scout is deliberately separated from the trading signal engine. Its job is
to discover useful techniques, compare them against our architecture, and
produce evidence-backed adoption proposals. It must never promote an idea to a
live trading rule without backtesting and review.
"""
from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


@dataclass(frozen=True, slots=True)
class ResearchFinding:
    title: str
    source: str
    category: str
    summary: str
    adoption: str
    confidence: float
    risk: str


@dataclass(frozen=True, slots=True)
class TechnologyScout:
    name: str = "BIST Technology Scout"
    mission: str = (
        "Continuously discover advances in systematic investing, agentic AI, "
        "alternative data, market microstructure, NLP and backtesting that can "
        "materially improve Bits100DEPO."
    )

    def research_rules(self) -> tuple[str, ...]:
        return (
            "Prefer primary sources, official documentation and peer-reviewed research.",
            "Separate proven production patterns from marketing claims.",
            "Check point-in-time availability, leakage and survivorship bias.",
            "Estimate BIST-specific feasibility, cost and data availability.",
            "Require an experiment/backtest before changing a live signal weight.",
            "Reject changes that increase complexity without measurable out-of-sample gain.",
        )


def build_scout_prompt(repository_summary: str, watchlist: str) -> str:
    """Build a provider-neutral prompt usable by Claude Agent SDK or another agent."""
    rules = "\n".join(f"- {rule}" for rule in TechnologyScout().research_rules())
    return dedent(
        f"""
        You are the BIST Technology Scout for Bits100DEPO.

        Mission: continuously find better research, data, modeling, agent and
        production patterns without blindly copying another firm's system.

        CURRENT SYSTEM:
        {repository_summary}

        WATCHLIST:
        {watchlist}

        RESEARCH RULES:
        {rules}
        Explicitly test for point-in-time integrity and look-ahead leakage in
        every proposed data source, feature, label and validation procedure.

        For every candidate improvement return:
        1. What changed in the world/research/tooling.
        2. Primary source URL and publication/release date.
        3. Why it may matter for BIST and pre-limit-up detection.
        4. Exact module or workflow it would improve.
        5. Estimated implementation effort: S/M/L.
        6. Expected benefit and measurable KPI.
        7. Failure modes: leakage, overfit, manipulation, latency, licensing, cost.
        8. Decision: ADOPT, EXPERIMENT, WATCH, or REJECT.

        Do not invent sources. Do not turn an interesting paper into a trading
        rule without an out-of-sample experiment. Rank findings by expected
        information value, not novelty.
        """
    ).strip()
