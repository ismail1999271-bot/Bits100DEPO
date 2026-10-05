import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ollama_task.py"
spec = importlib.util.spec_from_file_location("ollama_task", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_prompt_includes_rules_and_files():
    prompt = mod.build_prompt("add test", ["src/bist_hunter/risk.py"])
    assert "No automatic orders" in prompt and "FILE src/bist_hunter/risk.py" in prompt


def test_extract_diff():
    text = "plan\n```diff\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n```\n"
    assert mod.extract_diff(text).startswith("--- a/x")
    assert mod.extract_diff("no diff") is None
