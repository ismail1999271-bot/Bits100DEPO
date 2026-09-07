"""Optional agent layer for research and system improvement."""

from .quant_researcher import ExperimentSpec, QuantResearcher, build_quant_research_prompt
from .research_orchestrator import ResearchLoop
from .signal_auditor import AuditFinding, AuditReport, SignalAuditor, build_auditor_prompt
from .technology_scout import ResearchFinding, TechnologyScout, build_scout_prompt

__all__ = [
    "AuditFinding",
    "AuditReport",
    "ExperimentSpec",
    "QuantResearcher",
    "ResearchFinding",
    "ResearchLoop",
    "SignalAuditor",
    "TechnologyScout",
    "build_auditor_prompt",
    "build_quant_research_prompt",
    "build_scout_prompt",
]
