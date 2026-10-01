from datetime import datetime, timezone

import pandas as pd
import pytest

from bist_hunter.adapters import ProviderError
from bist_hunter.borsapy_adapter import BorsapyTransport, frame_to_rows
from bist_hunter.provider_status import BLOCKED, CONNECTED, SPECS, ProviderClient, status_board


def _frame(days):
    idx = pd.to_datetime(days)
    n = len(days)
    return pd.DataFrame({"Open": [10.0] * n, "High": [11.0] * n, "Low": [9.5] * n, "Close": [10.5] * n,
                         "Volume": [1000.0] * n}, index=idx)


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, **kw):
        if self.symbol == "BAD":
            raise RuntimeError("boom")
        return _frame(["2026-09-28", "2026-09-29", "2026-09-30"])


def test_today_bar_dropped_while_session_open():
    now = datetime(2026, 9, 30, 9, 0, tzinfo=timezone.utc)  # 12:00 Istanbul, session open
    rows = frame_to_rows("AAA", _frame(["2026-09-29", "2026-09-30", "2026-10-01"]), now)
    assert [r["timestamp"][:10] for r in rows] == ["2026-09-29"]


def test_today_bar_kept_after_close():
    now = datetime(2026, 9, 30, 16, 0, tzinfo=timezone.utc)  # 19:00 Istanbul
    rows = frame_to_rows("AAA", _frame(["2026-09-30"]), now)
    assert len(rows) == 1


def test_client_with_borsapy_backend_connected_and_skips_bad_symbol():
    now = datetime(2026, 10, 1, 7, 0, tzinfo=timezone.utc)
    transport = BorsapyTransport(FakeTicker, now=lambda: now)
    client = ProviderClient(env={"BIST_DATA_BACKEND": "borsapy"}, transport=transport)
    status, bars = client.fetch("ohlcv", {"symbols": "AAA,BAD", "start": "2026-09-01", "end": "2026-10-01"},
                                now=now)
    assert status.status == CONNECTED and len(bars) == 3
    assert "BAD" in transport.skipped and "gecikmeli" in status.reason


def test_all_failures_block():
    now = datetime(2026, 10, 1, 7, 0, tzinfo=timezone.utc)
    client = ProviderClient(env={"BIST_DATA_BACKEND": "borsapy"},
                            transport=BorsapyTransport(FakeTicker, now=lambda: now))
    status, bars = client.fetch("ohlcv", {"symbols": "BAD"}, now=now)
    assert status.status == BLOCKED and bars == []


def test_other_domains_rejected_and_board_reflects_backend():
    with pytest.raises(ProviderError):
        BorsapyTransport(FakeTicker)(SPECS["kap"], {"symbols": "A"})
    board = {s.domain: s for s in status_board({"BIST_DATA_BACKEND": "borsapy"})}
    assert board["ohlcv"].status == CONNECTED and board["kap"].status == "MISSING"
