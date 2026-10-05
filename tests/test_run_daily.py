import json
import subprocess
import sys


def test_run_daily_blocked_without_providers(tmp_path):
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": "src"}
    out = tmp_path / "snap.json"
    proc = subprocess.run([sys.executable, "scripts/run_daily.py", "--output", str(out), "--charts",
                           str(tmp_path / "c"), "--state", str(tmp_path / "s.json")],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    snap = json.loads(out.read_text(encoding="utf-8"))
    assert snap["top_quant_scores"]["status"] == "MISSING"
    assert snap["daily_plan"]["status"] == "OK"
    assert "status=BLOCKED" in proc.stdout
