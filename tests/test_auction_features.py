from datetime import datetime

import pandas as pd

from bist_hunter.auction_features import (
    TZ, AuctionObservation, build_auction_features, observation_from_record,
)
from bist_hunter.e2e_pipeline import row_components, run_e2e_backtest
from bist_hunter.synthetic import make_dataset

NOW = datetime(2026, 9, 11, 10, 0, tzinfo=TZ)


def _obs(minute, indicative, bid, ask, **extra):
    return AuctionObservation(
        "thyao", datetime(2026, 9, 11, 9, minute, tzinfo=TZ), indicative, 100.0, bid, ask, **extra
    )


def _full():
    flow = dict(new_buy_orders=40, new_sell_orders=20, cancelled_buy_orders=2, cancelled_sell_orders=6,
                bid_orders=100, ask_orders=60)
    return [
        _obs(40, 102.0, 1000, 800, matched_volume=0, **flow),
        _obs(45, 104.0, 1400, 800, matched_volume=0, **flow),
        _obs(50, 106.0, 2000, 700, matched_volume=0, **flow),
        _obs(55, 109.0, 3000, 600, matched_volume=0, **flow),
    ]


def test_complete_trajectory_features():
    result = build_auction_features(_full(), now=NOW)
    assert result.status == "COMPLETE"
    assert result.symbol == "THYAO"
    assert [s.slot for s in result.snapshots] == ["09:40", "09:45", "09:50", "09:55"]
    assert [(t.from_slot, t.to_slot) for t in result.transitions] == [
        ("09:40", "09:45"), ("09:45", "09:50"), ("09:50", "09:55")]
    assert result.transitions[0].indicative_change_delta == 2.0
    assert result.slope > 0
    assert result.acceleration == 1.0
    assert result.bid_pressure > result.ask_pressure
    assert result.order_arrival == 60
    assert result.cancellation_pressure < 0
    assert result.snapshots[-1].bid_ask_ratio == 5.0
    assert 50 < result.component_score <= 100
    assert "rising_trajectory" in result.reasons


def test_missing_slot_is_partial_and_not_interpolated():
    obs = _full()
    del obs[1]
    result = build_auction_features(obs, now=NOW)
    assert result.status == "PARTIAL"
    assert len(result.snapshots) == 3
    assert [(t.from_slot, t.to_slot) for t in result.transitions] == [("09:50", "09:55")]


def test_missing_fields_stay_none():
    obs = [AuctionObservation("AAA", datetime(2026, 9, 11, 9, 40, tzinfo=TZ), 101.0, 100.0)]
    result = build_auction_features(obs, now=NOW)
    snap = result.snapshots[0]
    assert snap.imbalance is None and snap.order_count is None and snap.volume is None
    assert result.cancellation_pressure is None and result.order_arrival is None
    assert result.component_score is not None


def test_fail_closed_cases():
    assert build_auction_features([], now=NOW).status == "BLOCKED_MISSING_AUCTION_DATA"
    future = build_auction_features(_full(), now=datetime(2026, 9, 11, 9, 50, tzinfo=TZ))
    assert future.status == "BLOCKED_FUTURE_TIMESTAMP"
    dup = _full() + [_obs(55, 109.0, 3000, 600)]
    assert build_auction_features(dup, now=NOW).status == "BLOCKED_DUPLICATE_EVENT"
    off = [_obs(42, 101.0, 10, 10)]
    assert build_auction_features(off, now=NOW).status == "BLOCKED_INVALID_TIMESTAMP"
    bad = [_obs(40, -1.0, 10, 10)]
    assert build_auction_features(bad, now=NOW).status == "BLOCKED_INVALID_PRICE"
    naive = [AuctionObservation("AAA", datetime(2026, 9, 11, 9, 40), 101.0, 100.0)]
    assert build_auction_features(naive, now=NOW).status == "BLOCKED_INVALID_TIMESTAMP"


def test_record_mapping_does_not_invent_values():
    obs = observation_from_record(
        {"observed_at": "2026-09-11T09:40:00+03:00", "auction_price": 10.5, "previous_close": 10}, symbol="aaa"
    )
    assert obs.indicative_price == 10.5 and obs.bid_qty is None and obs.new_buy_orders is None


def test_auction_component_feeds_quant_components_same_day_only():
    feats = build_auction_features(_full(), now=NOW)
    same = row_components("THYAO", pd.Timestamp("2026-09-11 18:10", tz=TZ), 0.4, None,
                          {("THYAO", "2026-09-11"): feats})
    other = row_components("THYAO", pd.Timestamp("2026-09-10 18:10", tz=TZ), 0.4, None,
                           {("THYAO", "2026-09-11"): feats})
    assert same["auction"] == feats.component_score and same["tavan_dna"] == 40.0
    assert "auction" not in other


def test_blocked_auction_is_not_used():
    blocked = build_auction_features([], now=NOW)
    comps = row_components("THYAO", pd.Timestamp("2026-09-11", tz=TZ), 0.4, None,
                           {("THYAO", "2026-09-11"): blocked})
    assert "auction" not in comps


def test_e2e_without_trajectories_fails_closed():
    frame = pd.DataFrame(make_dataset(days=80))
    assert run_e2e_backtest(frame) == ()
