#!/usr/bin/env python3
"""Delegate a coding/research task to a LOCAL Ollama model, safely.

    python scripts/ollama_task.py "Add ADX feature tests" --files src/bist_hunter/technical.py \
        --model qwen2.5-coder:14b

The model's answer is saved under artifacts/ollama/. If it contains a unified
diff, the diff is saved as a .patch and checked with `git apply --check`.
Nothing is applied or committed automatically: apply the patch yourself, then
run `pytest -q && ruff check . && python scripts/check_no_order_routing.py`.
AI output is a research/code proposal, never a trading signal (AGENTS.md).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RULES = (ROOT / "AGENTS.md").read_text(encoding="utf-8") if (ROOT / "AGENTS.md").exists() else ""


def build_prompt(task: str, files: list[str]) -> str:
    parts = [
        "You are a senior Python quant engineer working on the Bits100 BIST research platform.",
        "Follow these repository rules strictly:\n" + RULES,
        "Answer with a short plan and ONE unified diff (```diff ... ```) against the files below.",
        f"TASK:\n{task}",
    ]
    for name in files:
        path = ROOT / name
        parts.append(f"FILE {name}:\n```python\n{path.read_text(encoding='utf-8')}\n```")
    return "\n\n".join(parts)


def call_ollama(prompt: str, model: str, host: str, timeout: float) -> str:
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0.1}}).encode("utf-8")
    request = Request(f"{host.rstrip('/')}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))["response"]


def extract_diff(text: str) -> str | None:
    match = re.search(r"```(?:diff|patch)\n(.*?)```", text, flags=re.S)
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("--files", nargs="*", default=[])
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b"))
    parser.add_argument("--host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    try:
        answer = call_ollama(build_prompt(args.task, args.files), args.model, args.host, args.timeout)
    except (URLError, TimeoutError, KeyError) as exc:
        print(f"BLOCKED: Ollama not reachable at {args.host}: {exc}")
        return 2
    out = ROOT / "artifacts" / "ollama"
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / f"{stamp}.md").write_text(answer, encoding="utf-8")
    print(f"answer -> artifacts/ollama/{stamp}.md")
    diff = extract_diff(answer)
    if diff:
        patch = out / f"{stamp}.patch"
        patch.write_text(diff, encoding="utf-8")
        check = subprocess.run(["git", "apply", "--check", str(patch)], cwd=ROOT, capture_output=True, text=True)
        print(f"patch -> {patch.relative_to(ROOT)} | git apply --check: "
              f"{'OK' if check.returncode == 0 else 'FAILED ' + check.stderr.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
