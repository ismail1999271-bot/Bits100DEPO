"""Optional BorsaPy data backend (research / development use).

BorsaPy (https://github.com/saidsurucu/borsapy) serves TradingView data that is
~15 minutes delayed and is licensed for personal/educational use only; live
production use needs a separate Borsa Istanbul data licence. Selecting it is an
explicit opt-in: ``BIST_DATA_BACKEND=borsapy``.

Rules kept here
* Rows go through the normal OHLCV contract (``market_contracts.parse_ohlcv``).
* A bar of the *current* Istanbul session is dropped until the session is
  closed (>= 18:10), so a half-formed candle can never leak into a signal.
* A symbol that fails is skipped and reported; nothing is filled or invented.
* If the library is missing the backend raises ``ProviderError`` (-> BLOCKED).
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any, Callable

import pandas as pd

from .adapters import ProviderError
from .auction_features import TZ

BACKEND_ENV = "BIST_DATA_BACKEND"
SESSION_CLOSE = time(18, 10)


def _default_factory() -> Callable[[str], Any]:
    try:
        import borsapy
    except Exception as exc:  # ImportError or dependency errors
        raise ProviderError(f"borsapy is not usable: {exc}") from exc
    return borsapy.Ticker


def frame_to_rows(symbol: str, frame: pd.DataFrame, now: datetime) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    local_now = now.astimezone(TZ)
    session_closed = local_now.time() >= SESSION_CLOSE or local_now.weekday() >= 5
    rows = []
    for idx, rec in frame.iterrows():
        day = pd.Timestamp(idx)
        day = day.tz_localize(TZ) if day.tzinfo is None else day.tz_convert(TZ)
        day = day.normalize()
        # a bar stamped with today's date belongs to a session that may still be running
        if day.date() == local_now.date() and not session_closed:
            continue
        if day.date() > local_now.date():
            continue
        close_ts = day.replace(hour=18, minute=10).tz_convert(timezone.utc)
        rows.append({"symbol": symbol.upper(), "timestamp": close_ts.isoformat(), "interval": "1D",
                     "open": float(rec["Open"]), "high": float(rec["High"]), "low": float(rec["Low"]),
                     "close": float(rec["Close"]), "volume": float(rec["Volume"])})
    return rows


class BorsapyTransport:
    """Callable with the ``ProviderClient`` transport signature: (spec, params) -> records."""

    def __init__(self, ticker_factory: Callable[[str], Any] | None = None,
                 now: Callable[[], datetime] | None = None, max_workers: int = 8) -> None:
        self.max_workers = max_workers
        self._factory = ticker_factory
        self._now = now or (lambda: datetime.now(timezone.utc))
        self.skipped: dict[str, str] = {}

    def __call__(self, spec, params: dict[str, str]) -> list[dict[str, Any]]:
        if spec.domain != "ohlcv":
            raise ProviderError(f"borsapy backend does not serve domain {spec.domain}")
        factory = self._factory or _default_factory()
        symbols = [s for s in params.get("symbols", params.get("symbol", "")).split(",") if s]
        if not symbols:
            raise ProviderError("no symbols requested")
        now = self._now()
        rows: list[dict[str, Any]] = []
        self.skipped = {}
        def one(symbol: str):
            try:
                frame = factory(symbol).history(start=params.get("start"), end=params.get("end"), interval="1d")
                return frame_to_rows(symbol, frame, now)
            except Exception as exc:
                self.skipped[symbol] = str(exc)[:120]
                return []

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for part in pool.map(one, symbols):
                rows.extend(part)
        if not rows:
            raise ProviderError(f"borsapy returned no bars (skipped={len(self.skipped)})")
        return rows


def list_symbols(companies_fn=None) -> list[str]:
    """All currently listed BIST tickers from ``borsapy.companies()`` (live-view universe)."""
    try:
        if companies_fn is None:
            import borsapy

            companies_fn = borsapy.companies
        frame = companies_fn()
        symbols = [str(t).strip().upper() for t in frame["ticker"].tolist() if str(t).strip()]
    except Exception as exc:
        raise ProviderError(f"borsapy companies() failed: {exc}") from exc
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        raise ProviderError("borsapy returned an empty company list")
    return symbols
