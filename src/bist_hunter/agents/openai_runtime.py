"""OpenAI GPT-6 Astra runtime for read-only Bits100 research.

The runtime deliberately exposes research-only capabilities. Model output is
never treated as a trading instruction and never receives mutation/execution
authority. Authentication is supplied through ``OPENAI_API_KEY``.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass


DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_ENDPOINT = "https://api.openai.com/v1/responses"


@dataclass(frozen=True, slots=True)
class OpenAIResearchResult:
    model: str
    text: str
    response_id: str | None = None


def run_openai_research(
    prompt: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str = "high",
    max_output_tokens: int = 6000,
    timeout_seconds: int = 120,
) -> OpenAIResearchResult:
    """Run a read-only research request through the OpenAI Responses API.

    The request intentionally does not enable tools, shell, computer use,
    trading actions, or file mutation. Missing credentials fail closed.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise ValueError("unsupported reasoning_effort")
    if max_output_tokens < 256:
        raise ValueError("max_output_tokens must be >= 256")

    payload = {
        "model": model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        "input": [
            {
                "role": "developer",
                "content": (
                    "You are Bits100's senior quantitative research analyst. "
                    "You are research-only: do not place trades, change files, "
                    "invent market data, or present hypotheses as verified signals. "
                    "Separate facts, assumptions, hypotheses and required tests."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
    }

    request = urllib.request.Request(
        os.getenv("OPENAI_RESPONSES_URL", DEFAULT_ENDPOINT),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {detail[:1000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API connection failed: {exc.reason}") from exc

    text = str(body.get("output_text", "")).strip()
    if not text:
        chunks: list[str] = []
        for item in body.get("output", []):
            for content in item.get("content", []):
                value = content.get("text")
                if value:
                    chunks.append(str(value))
        text = "\n".join(chunks).strip()
    if not text:
        raise RuntimeError("OpenAI response contained no text output")

    return OpenAIResearchResult(
        model=str(body.get("model", payload["model"])),
        text=text,
        response_id=body.get("id"),
    )
