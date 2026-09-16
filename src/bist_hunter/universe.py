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
