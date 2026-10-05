"""Fail-closed loader for the live board. No source configured -> BLOCKED, never example data."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

import pandas as pd

from .adapters import ProviderError
from .live_board import alerts, build_board, fetch_quotes, summarize

LICENSE_NOTE = "BorsaPy/TradingView verisi ~15 dk gecikmelidir; kişisel/eğitim kullanımı içindir."


@dataclass(frozen=True, slots=True)
class LiveSnapshot:
    status: str  # OK / PARTIAL / BLOCKED
    reason: str
    fetched_at: datetime
    source: str
    board: pd.DataFrame | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    bars: pd.DataFrame | None = None
    skipped: dict[str, str] = field(default_factory=dict)
    alerts: list[dict[str, str]] = field(default_factory=list)
    note: str = ""


def _blocked(reason: str, now: datetime, source: str = "none") -> LiveSnapshot:
    return LiveSnapshot("BLOCKED", reason, now, source, summary={"status": "MISSING"})


def load_live_snapshot(
    env: Mapping[str, str] | None = None, *, now: datetime | None = None,
    symbols_fn: Callable[[], list[str]] | None = None,
    ticker_factory: Callable[[str], Any] | None = None,
    history_days: int = 150, bars: pd.DataFrame | None = None, fetch_history: bool = True,
) -> LiveSnapshot:
    now = now or datetime.now(timezone.utc)
    env = os.environ if env is None else env
    backend = env.get("BIST_DATA_BACKEND", "").strip().lower()
    if backend != "borsapy" and ticker_factory is None:
        return _blocked("Canlı kaynak yapılandırılmadı: BIST_DATA_BACKEND=borsapy (gecikmeli) ya da lisanslı "
                        "Level-1 sağlayıcı gerekir. Örnek veri gösterilmez.", now)
    try:
        if symbols_fn is None:
            from .real_adapters import load_symbol_universe

            symbols_fn = load_symbol_universe
        symbols = symbols_fn()
        if ticker_factory is None:
            from .borsapy_adapter import _default_factory

            ticker_factory = _default_factory()
    except ProviderError as exc:
        return _blocked(str(exc), now, backend or "custom")
    quotes, skipped = fetch_quotes(symbols, ticker_factory, now)
    if quotes.empty:
        return LiveSnapshot("BLOCKED", f"Hiç geçerli fiyat alınamadı ({len(skipped)} sembol atlandı)", now,
                            backend or "custom", skipped=skipped, summary={"status": "MISSING"})
    if bars is None and fetch_history:
        bars = _history(symbols, ticker_factory, now, history_days)
    board = build_board(quotes, bars, now=now)
    status = "OK" if len(skipped) <= 0.05 * len(symbols) else "PARTIAL"
    reason = f"{len(board)}/{len(symbols)} sembol" + (f", {len(skipped)} atlandı" if skipped else "")
    return LiveSnapshot(status, reason, now, backend or "custom", board, summarize(board), bars, skipped,
                        alerts(board), LICENSE_NOTE if backend == "borsapy" else "")


def _history(symbols: list[str], factory: Callable[[str], Any], now: datetime, days: int) -> pd.DataFrame | None:
    """Daily bars for indicators. A failure here only removes indicator columns; quotes still show."""
    from .borsapy_adapter import BorsapyTransport
    from .provider_status import SPECS

    start = (now - timedelta(days=days)).date().isoformat()
    transport = BorsapyTransport(factory, now=lambda: now, max_workers=12)
    try:
        rows = transport(SPECS["ohlcv"], {"symbols": ",".join(symbols), "start": start,
                                          "end": now.date().isoformat()})
    except ProviderError:
        return None
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame


def load_history_for_env(env: Mapping[str, str] | None = None, *, now: datetime | None = None,
                         days: int = 150) -> pd.DataFrame | None:
    """Daily history for the configured backend (None when not configured / unavailable)."""
    env = os.environ if env is None else env
    if env.get("BIST_DATA_BACKEND", "").strip().lower() != "borsapy":
        return None
    try:
        from .borsapy_adapter import _default_factory
        from .real_adapters import load_symbol_universe

        return _history(load_symbol_universe(), _default_factory(), now or datetime.now(timezone.utc), days)
    except ProviderError:
        return None
