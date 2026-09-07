"""Quant Researcher agent contract for hypothesis-to-experiment design.

This layer does not trade, change production weights, or claim performance. It
turns research ideas into reproducible experiment specifications that can be
run through the existing backtest/evaluation stack.
"""
from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """A reproducible, point-in-time-safe research experiment."""

    hypothesis: str
    target: str
    features: tuple[str, ...]
    baseline: str
    validation: str
    acceptance_criteria: tuple[str, ...]
    cost_model: str
    liquidity_filter: str
    leakage_controls: tuple[str, ...]
    experiment_id: str = ""


@dataclass(frozen=True, slots=True)
class QuantResearcher:
    name: str = "BIST Quant Researcher"
    mission: str = (
        "Convert promising research findings into small, falsifiable, reproducible "
        "experiments for BIST pre-limit-up ranking."
    )

    def research_rules(self) -> tuple[str, ...]:
        return (
            "Define one falsifiable hypothesis and one primary KPI before testing.",
            "Use point-in-time features and labels; forbid future information and look-ahead leakage.",
            "Use chronological walk-forward validation and preserve a final untouched holdout.",
            "Compare against a simple existing baseline; never compare only against a weak strawman.",
            "Include commission, slippage, spread/liquidity and position-size constraints.",
            "Track multiple-testing/experiment count and avoid selecting a model on the holdout.",
            "Report negative results and uncertainty; do not convert research into live rules directly.",
        )


def build_quant_research_prompt(
    finding: str,
    repository_summary: str,
    available_features: str,
) -> str:
    """Build a provider-neutral prompt for an agentic quant researcher."""
    rules = "\n".join(f"- {rule}" for rule in QuantResearcher().research_rules())
    return dedent(
        f"""
        You are the BIST Quant Researcher for Bits100DEPO.

        Convert the research finding below into ONE falsifiable experiment. Do not
        write production code, change signal weights, place orders, or claim that
        an untested idea is profitable.

        RESEARCH FINDING:
        {finding}

        CURRENT SYSTEM:
        {repository_summary}

        AVAILABLE FEATURES / DATA:
        {available_features}

        RESEARCH RULES:
        {rules}

        Return an ExperimentSpec containing:
        - hypothesis
        - target/label and forecast horizon
        - exact feature set and point-in-time availability
        - baseline model/rule
        - chronological walk-forward validation and final holdout
        - commission/slippage/spread cost model
        - liquidity and manipulation filters
        - explicit leakage controls
        - primary KPI plus secondary KPIs (Top-K precision, recall, hit-day rate,
          lead time, net P&L, drawdown)
        - minimum acceptance criteria
        - experiment_id and reproducibility requirements

        The result must be executable by the existing backtest/evaluation stack.
        Reject ideas that cannot be tested with available data.
        """
    ).strip()
