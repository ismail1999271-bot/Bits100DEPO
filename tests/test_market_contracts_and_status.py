from datetime import datetime, timedelta, timezone

import pytest

from bist_hunter import market_contracts as mc
from bist_hunter.provider_contracts import ContractError
from bist_hunter.provider_status import (
    BLOCKED, CONNECTED, INVALID, MISSING, SPECS, STALE, ProviderClient, evaluate_batch,
    format_status_board, signals_allowed, status_board,
)

NOW = datetime(2026, 9, 11, 7, 0, tzinfo=timezone.utc)


def iso(delta_seconds=0):
    return (NOW - timedelta(seconds=delta_seconds)).isoformat()


BOOK = {"symbol": "THYAO", "timestamp": iso(1), "level": "L2",
        "bids": [[300.0, 1000, 12], [299.75, 800, 9]], "asks": [[300.25, 500, 4], [300.5, 900, 7]]}


def test_ohlcv_contract():
    bar = mc.parse_ohlcv({"symbol": "thyao.is", "timestamp": iso(60), "open": 10, "high": 11, "low": 9.5,
                          "close": 10.5, "volume": 100}, now=NOW)
    assert bar.symbol == "THYAO" and bar.interval == "1D"
    with pytest.raises(ContractError):
        mc.parse_ohlcv({"symbol": "X", "timestamp": iso(), "open": 10, "high": 9, "low": 8, "close": 10,
                        "volume": 1})
    with pytest.raises(ContractError):  # future
        mc.parse_ohlcv({"symbol": "X", "timestamp": iso(-600), "open": 10, "high": 11, "low": 9, "close": 10,
                        "volume": 1}, now=NOW)
    with pytest.raises(ContractError):  # naive timestamp
        mc.parse_ohlcv({"symbol": "X", "timestamp": "2026-09-11T10:00:00", "open": 10, "high": 11, "low": 9,
                        "close": 10, "volume": 1})


def test_level1_and_book_contracts():
    assert mc.parse_level1({"symbol": "A", "timestamp": iso(), "bid": 10, "ask": 10.01, "bid_size": 5,
                            "ask_size": 7}, now=NOW).ask == 10.01
    with pytest.raises(ContractError):
        mc.parse_level1({"symbol": "A", "timestamp": iso(), "bid": 10, "ask": 10, "bid_size": 5, "ask_size": 7})
    book = mc.parse_order_book(BOOK, now=NOW)
    assert book.bids[0].orders == 12 and book.level == "L2"
    crossed = dict(BOOK, asks=[[299.0, 1]])
    with pytest.raises(ContractError):
        mc.parse_order_book(crossed)
    unsorted = dict(BOOK, bids=[[299.0, 1], [300.0, 1]])
    with pytest.raises(ContractError):
        mc.parse_order_book(unsorted)
    with pytest.raises(ContractError):
        mc.parse_order_book(dict(BOOK, level="L3"))


def test_order_event_tick_broker_institutional():
    ev = mc.parse_order_event({"symbol": "A", "timestamp": iso(), "order_id": "1", "side": "buy",
                               "action": "cancel", "quantity": 0}, now=NOW)
    assert ev.action == "CANCEL" and ev.price is None
    with pytest.raises(ContractError):
        mc.parse_order_event({"symbol": "A", "timestamp": iso(), "order_id": "1", "side": "BUY",
                              "action": "PLACE", "quantity": 1, "price": 1})
    assert mc.parse_tick({"symbol": "A", "timestamp": iso(), "price": 1, "quantity": 2, "trade_id": "t"}).price == 1
    row = {"symbol": "A", "published_at": iso(), "broker": "xyz", "buy_quantity": 10, "sell_quantity": 4,
           "source": "akd", "event_id": "e1"}
    assert mc.parse_broker_flow(row).broker == "XYZ"
    inst = mc.parse_institutional_flow({"symbol": "A", "observed_at": iso(), "investor_type": "foreign",
                                        "net_value": -5.0, "source": "s", "event_id": "i1"})
    assert inst.net_value == -5.0
    with pytest.raises(ContractError):
        mc.parse_many(mc.parse_tick, [{"symbol": "A", "timestamp": iso(), "price": 1, "quantity": 1,
                                       "trade_id": "t"}] * 2, key=lambda t: t.trade_id)


def test_status_board_blocks_and_missing():
    board = {s.domain: s for s in status_board({})}
    assert board["ohlcv"].status == BLOCKED
    assert board["fund_flow"].status == MISSING
    assert not signals_allowed(tuple(board.values()))
    env = {"BIST_MARKET_DATA_URL": "https://x", "BIST_LEVEL2_URL": "https://y"}
    board = {s.domain: s for s in status_board(env)}
    assert board["ohlcv"].status == BLOCKED  # no token
    assert "Level 2" in format_status_board(tuple(board.values()))


def test_evaluate_batch_statuses():
    spec = SPECS["level2"]
    status, parsed = evaluate_batch(spec, [BOOK], now=NOW)
    assert status.status == CONNECTED and len(parsed) == 1
    assert evaluate_batch(spec, [dict(BOOK, timestamp=iso(600))], now=NOW)[0].status == STALE
    assert evaluate_batch(spec, [dict(BOOK, asks=[[1.0, 1]])], now=NOW)[0].status == INVALID
    assert evaluate_batch(spec, [BOOK, BOOK], now=NOW)[0].status == INVALID
    assert evaluate_batch(spec, [], now=NOW)[0].status == MISSING


def test_client_uses_transport_only_when_credentialed():
    calls = []

    def transport(spec, params):
        calls.append(spec.domain)
        return {"data": [BOOK]}

    client = ProviderClient(env={}, transport=transport)
    assert client.fetch("level2", {"symbol": "THYAO"}, now=NOW)[0].status == MISSING
    assert calls == []
    env = {"BIST_LEVEL2_URL": "https://y", "BIST_LEVEL2_URL_TOKEN": "t"}
    status, rows = ProviderClient(env=env, transport=transport).fetch("level2", {}, now=NOW)
    assert status.status == CONNECTED and rows[0].symbol == "THYAO"
