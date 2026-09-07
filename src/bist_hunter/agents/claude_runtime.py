"""Optional Claude Agent SDK runtime for the technology scout.

The core package stays dependency-light. Install the optional ``agents`` extra
only where an Anthropic API key and the Claude Agent SDK are available.
"""
from __future__ import annotations

from pathlib import Path


async def run_claude_scout(prompt: str, project_root: str | Path = ".") -> str:
    """Run a read/research-only Claude scout and return its text report.

    The scout is intentionally denied file mutation and shell tools. Production
    code changes must go through the normal experiment/backtest/CI path.
    """
    try:
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Claude Agent SDK is not installed. Install the 'agents' optional extra."
        ) from exc

    options = ClaudeAgentOptions(
        cwd=str(Path(project_root).resolve()),
        system_prompt=(
            "You are a read-only R&D scout. Research and report; never modify files, "
            "execute shell commands, place trades, or change production configuration."
        ),
        tools=["WebSearch", "WebFetch", "Read", "Grep", "Glob"],
        allowed_tools=["WebSearch", "WebFetch", "Read", "Grep", "Glob"],
        max_turns=8,
    )

    chunks: list[str] = []
    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    chunks.append(block.text)
    return "\n".join(chunks).strip()
