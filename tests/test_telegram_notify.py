import inspect
from datetime import datetime, timedelta, timezone

import pytest

from bist_hunter import telegram_notify, telegram_report
from bist_hunter.fail_closed import GateInput
from bist_hunter.provider_status import status_board
from bist_hunter.research_ranking import SymbolResearch, rank_research
from bist_hunter.universe import build_bist100_plus_universe

NOW = datetime(2026, 9, 11, 7, 0, tzinfo=timezone.utc)


def test_outbound_only_no_command_or_order_path():
    source = inspect.getsource(telegram_notify) + inspect.getsource(telegram_report)
    for forbidden in ("getUpdates", "setWebhook", "CommandHandler", "create_order", "place_order"):
        assert forbidden not in source


def test_kinds_and_footer():
    with pytest.raises(ValueError):
        telegram_notify.format_message("ORDER", "x", "y")
    msg = telegram_notify.ci_status("feat/live-data-stack", "abcdef123", True)
    assert msg.startswith("[CI_STATUS]") and "Otomatik emir yoktur" in msg


def test_signal_and_status_reports():
    universe = build_bist100_plus_universe(["AAA", "BBB"], [], as_of="2026-09-11", source="t")
    gate = GateInput("AAA", NOW, NOW - timedelta(seconds=5), 60, None, provider_statuses={"ohlcv": "CONNECTED"})
    table = rank_research(universe, [SymbolResearch("AAA", gate, {"auction": 90, "tavan_dna": 80})])
    text = telegram_notify.signal_report(table)
    assert "AAA" in text and "BLOCKED sembol: 1" in text
    assert "⛔ BIST Market (OHLCV): BLOCKED" in telegram_notify.system_status(status_board({}))


def test_notify_chunks_and_unconfigured(monkeypatch):
    sent = []
    assert telegram_notify.notify("x\n" * 5000, sender=lambda c: sent.append(c) or True)
    assert len(sent) >= 2 and all(len(c) <= telegram_notify.MAX_LEN + 1 for c in sent)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert telegram_notify.notify("hello") is False
