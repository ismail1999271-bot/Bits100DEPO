import pytest

pytest.importorskip("streamlit")
from pathlib import Path  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "dashboard" / "live.py"
from streamlit.testing.v1 import AppTest  # noqa: E402

from bist_hunter import live_feed  # noqa: E402
from tests.test_live_board import NOW, FakeTicker  # noqa: E402


def _run():
    import streamlit as st

    st.cache_data.clear()
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_blocked_screen_without_source(monkeypatch):
    monkeypatch.delenv("BIST_DATA_BACKEND", raising=False)
    at = _run()
    assert not at.exception
    assert any("Canlı veri yok" in e.value for e in at.error)


def test_full_screen_with_injected_test_double(monkeypatch):
    real = live_feed.load_live_snapshot

    def fake(*a, **kw):
        return real({"BIST_DATA_BACKEND": "borsapy"}, now=NOW, symbols_fn=lambda: ["AAA", "BBB", "CCC"],
                    ticker_factory=FakeTicker)

    monkeypatch.setattr(live_feed, "load_live_snapshot", fake)
    monkeypatch.setattr(live_feed, "load_history_for_env", lambda *a, **k: None)
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 6
    assert len(at.metric) >= 6
    assert len(at.dataframe) >= 1
