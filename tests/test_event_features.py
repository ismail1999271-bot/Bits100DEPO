from datetime import datetime, timedelta, timezone

import pytest

from bist_hunter import market_contracts as mc
from bist_hunter.event_features import (
    MISSING_DATA, OK, broker_features, deduplicate, event_score, fund_flow_features,
    institutional_features, normalize_kap, normalize_news, quant_components,
)
from bist_hunter.provider_contracts import ContractError

AS_OF = datetime(2026, 9, 11, 6, 0, tzinfo=timezone.utc)


def t(hours):
    return (AS_OF - timedelta(hours=hours)).isoformat()


def test_kap_normalization_contract():
    ev = normalize_kap([{"symbol": "thyao.is", "id": "k1", "published_at": t(2), "subject": "Yeni iş ilişkisi",
                         "disclosure_type": "ODA"}])[0]
    assert (ev.symbol, ev.event_id, ev.source, ev.event_type, ev.headline) == (
        "THYAO", "k1", "KAP", "ODA", "Yeni iş ilişkisi")
    with pytest.raises(ContractError):
        normalize_kap([{"symbol": "A", "id": "1", "published_at": "2026-09-11T09:00:00", "title": "x"}])
    with pytest.raises(ContractError):
        normalize_news([{"symbol": "A", "event_id": "1", "published_at": t(1)}])


def test_dedup_by_id_and_cross_source_headline():
    rows = [
        {"symbol": "A", "event_id": "1", "published_at": t(5), "title": "A şirketi ihale kazandı", "source": "x"},
        {"symbol": "A", "event_id": "1", "published_at": t(5), "title": "A şirketi ihale kazandı", "source": "x"},
        {"symbol": "A", "event_id": "9", "published_at": t(4), "title": "A Şirketi, ihale kazandı!", "source": "y"},
        {"symbol": "A", "event_id": "5", "published_at": t(1), "title": "Başka haber", "source": "x"},
    ]
    kept = deduplicate(normalize_news(rows))
    assert [e.event_id for e in kept] == ["1", "5"]


def test_event_score_is_point_in_time_and_missing_is_none():
    events = normalize_kap([
        {"symbol": "A", "event_id": "1", "published_at": t(3), "title": "Yeni sözleşme imzalandı"},
        {"symbol": "A", "event_id": "2", "published_at": t(-2), "title": "İflas başvurusu"},  # future
    ])
    score = event_score(events, "A", AS_OF)
    assert score.status == OK and score.events == 1 and score.score > 50
    missing = event_score(events, "B", AS_OF)
    assert missing.status == MISSING_DATA and missing.score is None


def test_fund_flow_features():
    rows = [
        {"symbol": "A", "fund_code": "F1", "net_flow": 30e6, "observed_at": t(10)},
        {"symbol": "A", "fund_code": "F2", "net_flow": -10e6, "observed_at": t(10)},
        {"symbol": "A", "fund_code": "F3", "net_flow": 99e9, "observed_at": t(-1)},  # future ignored
    ]
    f = fund_flow_features(rows, "A", AS_OF)
    assert (f.inflow, f.outflow, f.net_flow, f.funds) == (30e6, 10e6, 20e6, 2)
    assert f.concentration == pytest.approx(0.75 ** 2 + 0.25 ** 2)
    assert f.score > 50
    assert fund_flow_features(rows, "B", AS_OF).status == MISSING_DATA


def test_broker_and_institutional_features():
    brokers = [mc.parse_broker_flow({"symbol": "A", "published_at": t(1), "broker": b, "buy_quantity": bq,
                                     "sell_quantity": sq, "source": "akd", "event_id": b})
               for b, bq, sq in (("X", 1000, 100), ("Y", 200, 300), ("Z", 0, 400))]
    bf = broker_features(brokers, "A", AS_OF)
    assert (bf.broker_buying, bf.broker_selling, bf.net_broker_flow) == (1200, 800, 400)
    assert bf.brokers == 3 and bf.score > 50
    assert broker_features(brokers, "A", AS_OF - timedelta(days=1)).status == MISSING_DATA
    inst = [mc.parse_institutional_flow({"symbol": "A", "observed_at": t(24 * d), "investor_type": "foreign",
                                         "net_value": v, "source": "s", "event_id": str(d)})
            for d, v in ((1, 5e6), (2, 3e6), (3, -1e6))]
    f = institutional_features(inst, "A", AS_OF)
    assert f.institutional_flow == 7e6 and f.accumulation_days == 2 and f.observation_days == 3
    comps = quant_components(broker=bf, institutional=f, fund=fund_flow_features([], "A", AS_OF))
    assert set(comps) == {"broker", "institutional"}
