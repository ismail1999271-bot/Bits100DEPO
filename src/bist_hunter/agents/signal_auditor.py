"""Adversarial audit contract for research experiments.

The Signal Auditor is intentionally hostile to attractive backtest results. It
checks common sources of false edge before an experiment can move toward model
selection or production review.
"""
from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

from .quant_researcher import ExperimentSpec


@dataclass(frozen=True, slots=True)
class AuditFinding:
    check: str
    passed: bool
    severity: str
    detail: str


@dataclass(frozen=True, slots=True)
class AuditReport:
    passed: bool
    findings: tuple[AuditFinding, ...]

    @property
    def blocking_findings(self) -> tuple[AuditFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if not finding.passed and finding.severity == "BLOCKER"
        )


@dataclass(frozen=True, slots=True)
class SignalAuditor:
    name: str = "BIST Signal Auditor"
    mission: str = (
        "Attack BIST trading experiments for leakage, overfit, survivorship, "
        "cost, liquidity, manipulation and regime fragility before adoption."
    )

    def audit(self, experiment: ExperimentSpec) -> AuditReport:
        """Run deterministic contract checks; no market data or trading is touched."""
        findings: list[AuditFinding] = []

        def check(name: str, passed: bool, detail: str, severity: str = "BLOCKER") -> None:
            findings.append(AuditFinding(name, passed, severity, detail))

        check(
            "hypothesis",
            bool(experiment.hypothesis.strip()),
            "Hypothesis must be explicit and falsifiable.",
        )
        check("target", bool(experiment.target.strip()), "Target and horizon must be defined.")
        check("features", bool(experiment.features), "At least one feature must be declared.")
        check("baseline", bool(experiment.baseline.strip()), "A real baseline is required.")
        check(
            "walk-forward",
            "walk" in experiment.validation.lower() and "chron" in experiment.validation.lower(),
            "Validation must use chronological walk-forward testing.",
        )
        check(
            "holdout",
            "holdout" in experiment.validation.lower(),
            "A final untouched holdout must be reserved.",
        )
        check(
            "cost-model",
            bool(experiment.cost_model.strip())
            and any(term in experiment.cost_model.lower() for term in ("commission", "slippage", "spread")),
            "Costs must include commission plus realistic execution friction.",
        )
        check(
            "liquidity",
            bool(experiment.liquidity_filter.strip()),
            "Illiquid names need an explicit filter or sizing constraint.",
        )
        leakage_text = " ".join(experiment.leakage_controls).lower()
        check(
            "point-in-time/leakage",
            bool(experiment.leakage_controls)
            and any(term in leakage_text for term in ("point-in-time", "point in time", "look-ahead", "future")),
            "Point-in-time availability and look-ahead prevention must be explicit.",
        )
        criteria = " ".join(experiment.acceptance_criteria).lower()
        check(
            "out-of-sample-kpi",
            "out-of-sample" in criteria or "oos" in criteria,
            "Acceptance criteria must require out-of-sample evidence.",
        )
        check(
            "multiple-testing",
            any(term in criteria for term in ("multiple", "experiment count", "data snooping")),
            "Experiment selection must account for multiple testing/data snooping.",
            severity="MAJOR",
        )

        passed = not any(not finding.passed and finding.severity == "BLOCKER" for finding in findings)
        return AuditReport(passed=passed, findings=tuple(findings))


def build_auditor_prompt(experiment: str) -> str:
    """Build a provider-neutral adversarial review prompt."""
    return dedent(
        f"""
        You are the BIST Signal Auditor for Bits100DEPO.

        Audit this proposed experiment adversarially. You are not allowed to
        approve it merely because the backtest return is attractive.

        EXPERIMENT:
        {experiment}

        Check explicitly for:
        1. point-in-time violations and look-ahead leakage;
        2. survivorship and delisting bias;
        3. random/non-chronological validation or holdout contamination;
        4. data snooping, multiple testing and overfit;
        5. unrealistic fills, commission, spread and slippage;
        6. illiquidity, concentration and position-size assumptions;
        7. pump/manipulation sensitivity and abnormal-volume traps;
        8. regime dependence and instability across market/sector regimes;
        9. label leakage from news, KAP, fund-flow or social timestamps;
        10. reproducibility and missing-data failures.

        Return PASS/FAIL per check, severity, evidence, and a remediation plan.
        Any unresolved blocker means FAIL. Never promote a failed experiment to
        a live trading rule.
        """
    ).strip()
