"""Single-universe construction for the BIST100+ research scan."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Universe:
    name: str
    symbols: tuple[str, ...]
    as_of: str
    source: str


def build_bist100_plus_universe(
    bist100_symbols: Iterable[str],
    provider_symbols: Iterable[str],
    *,
    as_of: str,
    source: str,
) -> Universe:
    """Union BIST100 membership with all provider-covered symbols.

    Membership is supplied by an authoritative, point-in-time source. No
    hardcoded current constituent list is embedded in the repository, which
    avoids survivorship bias and stale index membership.
    """
    symbols = {str(symbol).strip().upper() for symbol in bist100_symbols if str(symbol).strip()}
    symbols.update(str(symbol).strip().upper() for symbol in provider_symbols if str(symbol).strip())
    if not symbols:
        raise ValueError("empty BIST100+ universe")
    if not as_of.strip() or not source.strip():
        raise ValueError("universe requires point-in-time as_of and source")
    return Universe("BIST100+", tuple(sorted(symbols)), as_of.strip(), source.strip())


def scan_universe(universe: Universe, ranker, *, max_candidates: int | None = None):
    """Run one ranker over the complete universe, then apply one global cap."""
    results = ranker(universe.symbols)
    ordered = tuple(results)
    if max_candidates is not None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        ordered = ordered[:max_candidates]
    return ordered


# ---------------------------------------------------------------------------
# Point-in-time membership (survivorship-bias safe) and the single entry point.
# ---------------------------------------------------------------------------
from datetime import date as _date  # noqa: E402


def _to_date(value) -> _date:
    if isinstance(value, _date):
        return value
    return _date.fromisoformat(str(value)[:10])


@dataclass(frozen=True, slots=True)
class Membership:
    """A symbol belongs to the universe on [valid_from, valid_to]; valid_to=None means still listed."""

    symbol: str
    valid_from: _date
    valid_to: _date | None = None

    def active(self, day: _date) -> bool:
        return self.valid_from <= day and (self.valid_to is None or day <= self.valid_to)


@dataclass(frozen=True, slots=True)
class UniverseHistory:
    memberships: tuple[Membership, ...]
    source: str

    def symbols_on(self, day) -> tuple[str, ...]:
        d = _to_date(day)
        return tuple(sorted({m.symbol for m in self.memberships if m.active(d)}))

    def as_of(self, day) -> Universe:
        symbols = self.symbols_on(day)
        if not symbols:
            raise ValueError(f"empty BIST100+ universe on {day}")
        return Universe("BIST100+", symbols, _to_date(day).isoformat(), self.source)

    def all_symbols(self) -> tuple[str, ...]:
        """Every symbol that was ever a member (incl. delisted) - use for backtests."""
        return tuple(sorted({m.symbol for m in self.memberships}))


def universe_history_from_records(records: Iterable[dict], *, source: str) -> UniverseHistory:
    """Build history from provider rows: symbol, valid_from|listed_at, valid_to|delisted_at."""
    memberships: list[Membership] = []
    for row in records:
        symbol = str(row.get("symbol", "")).strip().upper().removesuffix(".IS")
        start = row.get("valid_from", row.get("listed_at"))
        if not symbol or start in (None, ""):
            raise ValueError(f"universe record missing symbol/valid_from: {row}")
        end = row.get("valid_to", row.get("delisted_at"))
        m = Membership(symbol, _to_date(start), None if end in (None, "") else _to_date(end))
        if m.valid_to is not None and m.valid_to < m.valid_from:
            raise ValueError(f"valid_to before valid_from for {symbol}")
        memberships.append(m)
    if not memberships:
        raise ValueError("empty universe history")
    return UniverseHistory(tuple(memberships), source)


def filter_point_in_time(frame, history: UniverseHistory):
    """Keep only rows whose symbol was a member on that row's date.

    Delisted symbols stay in the sample for the dates they were members, so the
    backtest is not survivorship-biased; symbols listed later never leak back.
    """
    import pandas as pd

    dates = pd.to_datetime(frame["timestamp"]).dt.date
    symbols = frame["symbol"].astype(str).str.upper()
    keep = [
        any(m.symbol == s and m.active(d) for m in history.memberships)
        for s, d in zip(symbols, dates)
    ]
    return frame.loc[keep].copy()


def load_bist100_plus_universe(as_of=None) -> Universe:
    """THE single BIST100+ universe used by every live pipeline.

    Source priority: BIST_UNIVERSE_URL (provider) -> BIST_SYMBOLS. BIST100
    membership from BIST100_MEMBERSHIP_URL is unioned in when configured.
    Raises ProviderError (-> BLOCKED) when nothing is configured.
    """
    import os
    from datetime import UTC, datetime

    from .real_adapters import fetch_json_endpoint, load_symbol_universe

    day = _to_date(as_of) if as_of is not None else datetime.now(UTC).date()
    provider_symbols = load_symbol_universe()
    bist100: list[str] = []
    membership_url = os.getenv("BIST100_MEMBERSHIP_URL", "").strip()
    source = "BIST_UNIVERSE_URL" if os.getenv("BIST_UNIVERSE_URL") else "BIST_SYMBOLS"
    if membership_url:
        history = universe_history_from_records(
            fetch_json_endpoint(membership_url, "BIST_DATA_API_KEY"), source="BIST100_MEMBERSHIP_URL"
        )
        bist100 = list(history.symbols_on(day))
        source += "+BIST100_MEMBERSHIP_URL"
    return build_bist100_plus_universe(bist100, provider_symbols, as_of=day.isoformat(), source=source)
