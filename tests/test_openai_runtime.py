import json

import pytest

from bist_hunter.agents.openai_runtime import run_openai_research


def test_openai_runtime_fails_closed_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        run_openai_research("test")


def test_openai_runtime_rejects_invalid_reasoning():
    with pytest.raises(ValueError, match="reasoning_effort"):
        run_openai_research("test", api_key="dummy", reasoning_effort="none")


def test_openai_payload_contract(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"id": "r1", "model": "gpt-6-astra", "output_text": "ok"}).encode()

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return FakeResponse()

    import bist_hunter.agents.openai_runtime as runtime

    monkeypatch.setattr(runtime.urllib.request, "urlopen", fake_urlopen)
    result = run_openai_research("find hypotheses", api_key="secret")

    assert result.text == "ok"
    assert captured["body"]["model"] == "gpt-6-astra"
    assert captured["body"]["reasoning"]["effort"] == "high"
    assert "tools" not in captured["body"]
