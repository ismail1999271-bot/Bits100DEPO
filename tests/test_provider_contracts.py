from datetime import datetime, timezone

import pytest

from bist_hunter.provider_contracts import ContractError, parse_event_observation, parse_market_observation, validate_unique_keys


def test_market_contract_requires_timezone_and_positive_price():
    row = {
        "symbol": "TEST",
        "observed_at": "2026-09-16T06:40:00+00:00",
        "indicative_price": 100,
        "reference_price": 95,
        "bid_qty": 1000,
        "ask_qty": 500,
    }
    result = parse_market_observation(row, now=datetime(2026, 9, 16, 6, 40, tzinfo=timezone.utc), max_age_seconds=1)
    assert result.symbol == "TEST"
    with pytest.raises(ContractError):
        parse_market_observation({**row, "indicative_price": 0})


def test_duplicate_market_observation_is_rejected():
    row = {"symbol": "TEST", "observed_at": "2026-09-16T06:40:00+00:00"}
    with pytest.raises(ContractError):
        validate_unique_keys([row, row], ("symbol", "observed_at"))


def test_event_contract_is_point_in_time_and_identifiable():
    result = parse_event_observation({"symbol": "TEST", "published_at": "2026-09-16T06:40:00+00:00", "source": "KAP", "event_id": "42", "text": "event"})
    assert result.symbol == "TEST"
    assert result.event_id == "42"
