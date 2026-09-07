#!/usr/bin/env python3
"""Autonomous GPT-6 Astra coding loop for Bits100.

The agent audits the checked-out repository, proposes a unified diff, applies it,
runs the deterministic test/lint gates, and asks GPT-6 Astra to repair failures.
It never receives trading credentials or production execution authority.
"""
from __future__ import annotations

import os
import re
import subprocess
import urllib.error
import urllib.request
import json
from pathlib import Path

MODEL = os.getenv("OPENAI_MODEL", "gpt-6-astra")
ENDPOINT = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")
MAX_CONTEXT_BYTES = 900_000
MAX_ATTEMPTS = int(os.getenv("ASTRA_MAX_ATTEMPTS", "3"))

DENY_NAMES = {".env", ".env.local", ".env.production"}
DENY_PARTS = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", "node_modules"}
ALLOWED_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".json"}


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return proc.returncode, (proc.stdout + "\n" + proc.stderr).strip()


def repo_snapshot(root: Path) -> str:
    code, output = run(["git", "ls-files"], root)
    if code:
        raise RuntimeError(output)
    chunks: list[str] = []
    used = 0
    for raw_path in output.splitlines():
        path = Path(raw_path)
        if path.name in DENY_NAMES or any(part in DENY_PARTS for part in path.parts):
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        block = f"\n===== {raw_path} =====\n{text}\n"
        if used + len(block.encode("utf-8")) > MAX_CONTEXT_BYTES:
            continue
        chunks.append(block)
        used += len(block.encode("utf-8"))
    return "".join(chunks)


def call_astra(prompt: str, api_key: str) -> str:
    payload = {
        "model": MODEL,
        "input": [
            {
                "role": "developer",
                "content": (
                    "You are the senior software engineer responsible for completing Bits100. "
                    "Inspect the supplied repository snapshot and make concrete, production-quality "
                    "software improvements. Preserve fail-closed financial safety: never add live trading, "
                    "credential exfiltration, synthetic market data, look-ahead leakage, or bypasses of licensed data. "
                    "Return ONLY a unified git diff inside a single ```diff fence. No prose."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "reasoning": {"effort": "high"},
        "max_output_tokens": 30000,
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {detail[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API connection failed: {exc.reason}") from exc

    text = str(body.get("output_text", "")).strip()
    if not text:
        for item in body.get("output", []):
            for content in item.get("content", []):
                if content.get("text"):
                    text += str(content["text"])
    if not text.strip():
        raise RuntimeError("GPT-6 Astra returned no text")
    return text.strip()


def extract_diff(text: str) -> str:
    match = re.search(r"```diff\s*(.*?)\s*```", text, flags=re.DOTALL)
    diff = match.group(1).strip() if match else text.strip()
    if not diff.startswith("diff --git "):
        raise RuntimeError("GPT-6 Astra did not return a valid unified git diff")
    for line in diff.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            candidate = line[6:]
            path = Path(candidate)
            if candidate and (path.is_absolute() or ".." in path.parts or path.name in DENY_NAMES):
                raise RuntimeError(f"unsafe patch path: {candidate}")
    return diff + "\n"


def apply_diff(root: Path, diff: str) -> None:
    patch = root / ".astra.patch"
    patch.write_text(diff, encoding="utf-8")
    try:
        code, output = run(["git", "apply", "--check", str(patch)], root)
        if code:
            raise RuntimeError(f"patch check failed:\n{output}")
        code, output = run(["git", "apply", str(patch)], root)
        if code:
            raise RuntimeError(f"patch apply failed:\n{output}")
    finally:
        patch.unlink(missing_ok=True)


def quality_gate(root: Path) -> tuple[bool, str]:
    commands = [
        ["python", "-m", "compileall", "-q", "src", "tests"],
        ["python", "-m", "pytest", "-q"],
        ["ruff", "check", "."],
    ]
    reports: list[str] = []
    for cmd in commands:
        code, output = run(cmd, root)
        reports.append(f"$ {' '.join(cmd)}\n{output}")
        if code:
            return False, "\n\n".join(reports)
    return True, "\n\n".join(reports)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    snapshot = repo_snapshot(root)
    failures = "No test/lint failures yet. First perform a full architecture audit."
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = f"""BITS100 AUTONOMOUS COMPLETION PASS {attempt}/{MAX_ATTEMPTS}

Mission:
1. Audit the repository for missing or incomplete software needed for a production-ready research platform.
2. Prioritize real BIST/KAP/fund-flow/broker/institutional adapters, point-in-time correctness, ranking/scoring,
   Tavan-DNA historical training, walk-forward validation, paper trading, research-agent feedback, observability,
   CI/CD and production gates.
3. Implement the highest-value missing pieces now. Do not merely describe them.
4. Add or update tests for every behavioral change.
5. Do not remove existing safety gates just to make tests pass.
6. If external credentials/data contracts are unavailable, implement the adapter contract and fail-closed readiness
   behavior rather than inventing data.

PREVIOUS GATE RESULT:
{failures}

REPOSITORY SNAPSHOT:
{snapshot}
"""
        diff = extract_diff(call_astra(prompt, api_key))
        apply_diff(root, diff)
        ok, report = quality_gate(root)
        failures = report
        if ok:
            (root / "astra_completion_report.md").write_text(
                "# GPT-6 Astra Completion\n\n"
                f"Pass: {attempt}\n\n"
                "All compile, pytest and Ruff gates passed.\n\n"
                "The changes remain research/paper-trading safe; external credentials and licensed data "
                "providers are still required before production deployment.\n\n"
                "## Verification\n\n" + report + "\n",
                encoding="utf-8",
            )
            print(report)
            return 0
        # Restore only the failed pass before asking Astra for a cleaner repair.
        run(["git", "reset", "--hard", "HEAD"], root)
        run(["git", "clean", "-fd", "--exclude=.git"], root)
    print(failures)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
