import json
from datetime import datetime, timedelta, timezone

from bist_hunter.dashboard_data import blocked_snapshot, build_snapshot, market_status, write_snapshot
from bist_hunter.fail_closed import GateInput
from bist_hunter.provider_status import status_board
from bist_hunter.research_ranking import SymbolResearch, rank_research
from bist_hunter.universe import build_bist100_plus_universe

NOW = datetime(2026, 9, 11, 7, 0, tzinfo=timezone.utc)  # 10:00 Istanbul


def test_market_status_sessions():
    assert market_status(datetime(2026, 9, 11, 6, 45, tzinfo=timezone.utc))["phase"] == "OPENING_AUCTION"
    assert market_status(NOW)["phase"] == "CONTINUOUS"
    assert market_status(datetime(2026, 9, 12, 8, 0, tzinfo=timezone.utc))["phase"] == "CLOSED_WEEKEND"


def test_blocked_snapshot_has_no_numbers(tmp_path):
    snap = blocked_snapshot(NOW, status_board({}))
    assert snap["provider_status"]["status"] == "OK"
    for key in ("universe", "top_quant_scores", "auction_leaders", "paper_trading", "risk"):
        assert snap[key]["status"] == "MISSING" and snap[key]["data"] is None
    path = write_snapshot(snap, tmp_path / "s.json")
    assert json.loads(path.read_text())["disclaimer"]


def test_snapshot_from_ranking():
    universe = build_bist100_plus_universe(["AAA", "BBB"], [], as_of="2026-09-11", source="t")
    gate = GateInput("AAA", NOW, NOW - timedelta(seconds=5), 60, None, provider_statuses={"ohlcv": "CONNECTED"})
    ranking = rank_research(universe, [SymbolResearch("AAA", gate, {"auction": 90, "tavan_dna": 80},
                                                      relative_volume=3.0)])
    snap = build_snapshot(now=NOW, universe=universe, ranking=ranking)
    assert snap["universe"]["data"]["size"] == 2
    assert snap["top_quant_scores"]["data"][0]["Symbol"] == "AAA"
    assert snap["auction_leaders"]["data"][0]["Auction"] == 90
    assert snap["data_quality"]["data"] == {"ranked": 2, "blocked": 1}
