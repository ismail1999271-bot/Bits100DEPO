"""Research loop coordinating Scout -> Quant Researcher -> Signal Auditor."""
from __future__ import annotations

from dataclasses import dataclass

from .quant_researcher import ExperimentSpec, QuantResearcher, build_quant_research_prompt
from .signal_auditor import AuditReport, SignalAuditor, build_auditor_prompt
from .technology_scout import TechnologyScout, build_scout_prompt


@dataclass(frozen=True, slots=True)
class ResearchLoop:
    """Provider-neutral orchestration contract.

    The loop produces research artifacts only. Promotion into production remains
    a separate model-selection/review decision after real OOS evidence exists.
    """

    scout: TechnologyScout = TechnologyScout()
    quant: QuantResearcher = QuantResearcher()
    auditor: SignalAuditor = SignalAuditor()

    def scout_prompt(self, repository_summary: str, watchlist: str) -> str:
        return build_scout_prompt(repository_summary, watchlist)

    def quant_prompt(self, finding: str, repository_summary: str, available_features: str) -> str:
        return build_quant_research_prompt(finding, repository_summary, available_features)

    def audit_prompt(self, experiment: str) -> str:
        return build_auditor_prompt(experiment)

    def deterministic_audit(self, experiment: ExperimentSpec) -> AuditReport:
        return self.auditor.audit(experiment)
