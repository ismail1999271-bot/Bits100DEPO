from datetime import datetime, timedelta, timezone

import pandas as pd

from bist_hunter.provider_status import ProviderClient
from bist_hunter.research_run import run_research
from bist_hunter.synthetic import make_dataset  # tests only - never production

NOW = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)


def test_blocked_without_real_providers(monkeypatch):
    monkeypatch.delenv("BIST_SYMBOLS", raising=False)
    run = run_research(now=NOW, client=ProviderClient(env={}))
    assert run.status == "BLOCKED" and run.ranking is None
    assert run.snapshot["top_quant_scores"]["status"] == "MISSING"


def test_end_to_end_with_injected_transport(monkeypatch):
    monkeypatch.setenv("BIST_SYMBOLS", "AAA,BBB,CCC,DDD")
    monkeypatch.delenv("BIST_UNIVERSE_URL", raising=False)
    monkeypatch.delenv("BIST100_MEMBERSHIP_URL", raising=False)
    rows = make_dataset(symbols=("AAA", "BBB", "CCC"), days=120)
    shift = NOW - timedelta(hours=2) - rows[-1]["timestamp"]
    for r in rows:
        r["timestamp"] = (r["timestamp"] + shift).isoformat()

    def transport(spec, params):
        return {"data": rows} if spec.domain == "ohlcv" else {"data": []}

    env = {"BIST_MARKET_DATA_URL": "https://vendor", "BIST_MARKET_DATA_URL_TOKEN": "t"}
    run = run_research(now=NOW, client=ProviderClient(env=env, transport=transport))
    assert run.status == "OK"
    table = run.ranking
    assert list(table["Symbol"])[-1] == "DDD"  # no data -> BLOCKED, listed last
    assert table.loc[table["Symbol"] == "DDD", "Status"].item() == "BLOCKED"
    live = table[table["Symbol"] != "DDD"]
    assert live["Tavan-DNA"].notna().all() and live["Technical"].notna().all()
    assert live["Entry"].notna().all()
    assert run.snapshot["universe"]["data"]["size"] == 4
    assert isinstance(table, pd.DataFrame)


def test_charts_written_for_ranked_symbols(monkeypatch, tmp_path):
    monkeypatch.setenv("BIST_SYMBOLS", "AAA,BBB")
    monkeypatch.delenv("BIST_UNIVERSE_URL", raising=False)
    monkeypatch.delenv("BIST100_MEMBERSHIP_URL", raising=False)
    rows = make_dataset(symbols=("AAA", "BBB"), days=120)
    shift = NOW - timedelta(hours=2) - rows[-1]["timestamp"]
    for r in rows:
        r["timestamp"] = (r["timestamp"] + shift).isoformat()

    def transport(spec, params):
        return {"data": rows} if spec.domain == "ohlcv" else {"data": []}

    env = {"BIST_MARKET_DATA_URL": "https://vendor", "BIST_MARKET_DATA_URL_TOKEN": "t"}
    run = run_research(now=NOW, client=ProviderClient(env=env, transport=transport),
                       chart_dir=str(tmp_path), chart_top=5)
    assert run.status == "OK"
    sym = next(s for s, d in run.snapshot["symbols"].items() if d["Chart"]["status"] == "OK")
    assert (tmp_path / f"{sym}.png").exists()
