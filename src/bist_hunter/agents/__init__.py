"""Optional agent layer for research and system improvement."""

from .technology_scout import ResearchFinding, TechnologyScout, build_scout_prompt

__all__ = ["ResearchFinding", "TechnologyScout", "build_scout_prompt"]
