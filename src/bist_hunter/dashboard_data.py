"""Dashboard snapshot builder (pure data; the Streamlit UI only renders it).

The snapshot is assembled from real pipeline outputs. When an input is not
available the section carries status BLOCKED / MISSING and no numbers, so the
dashboard can never display fabricated values.
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any

import pandas as pd

from .auction_features import TZ

SESSIONS = (
    (time(9, 40), time(9, 55), "OPENING_AUCTION"),
    (time(9, 55), time(10, 0), "OPENING_MATCH"),
    (time(10, 0), time(18, 0), "CONTINUOUS"),
    (time(18, 0), time(18, 10), "CLOSING_AUCTION"),
)


def market_status(now: datetime) -> dict[str, str]:
    """Session by clock only; exchange holidays are NOT checked (flagged in the note)."""
    local = now.astimezone(TZ)
    if local.weekday() >= 5:
        phase = "CLOSED_WEEKEND"
    else:
        phase = "CLOSED"
        for start, end, name in SESSIONS:
            if start <= local.time() < end:
                phase = name
                break
    return {"phase": phase, "local_time": local.isoformat(timespec="minutes"),
            "note": "Resmi tatil takvimi kontrol edilmez / holiday calendar not checked"}


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return json.loads(value.to_json(orient="records", date_format="iso"))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


def _section(data: Any, missing_reason: str) -> dict[str, Any]:
    if data is None or (isinstance(data, (list, dict, tuple)) and not data) or \
            (isinstance(data, pd.DataFrame) and data.empty):
        return {"status": "MISSING", "reason": missing_reason, "data": None}
    return {"status": "OK", "data": _plain(data)}


def _leaders(ranking: pd.DataFrame | None, column: str, n: int = 10):
    if ranking is None or ranking.empty or column not in ranking.columns:
        return None
    ok = ranking[ranking["Status"] != "BLOCKED"].dropna(subset=[column])
    if ok.empty:
        return None
    return ok.sort_values(column, ascending=False).head(n)[["Symbol", column]]


def build_snapshot(
    *,
    now: datetime,
    universe=None,
    provider_statuses=None,
    ranking: pd.DataFrame | None = None,
    auction: dict[str, Any] | None = None,
    events: list[Any] | None = None,
    paper_summary: dict[str, Any] | None = None,
    paper_entries: list[Any] | None = None,
    symbol_details: dict[str, dict[str, Any]] | None = None,
    data_quality: dict[str, Any] | None = None,
    daily_plan: Any = None,
    stress: Any = None,
    overfit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ranking_ok = ranking is not None and not ranking.empty
    blocked = int((ranking["Status"] == "BLOCKED").sum()) if ranking_ok else None
    return {
        "generated_at": now.isoformat(),
        "disclaimer": "Araştırma platformudur; yatırım tavsiyesi değildir. Otomatik emir yoktur.",
        "market_status": market_status(now),
        "provider_status": _section(provider_statuses, "provider registry not evaluated"),
        "data_quality": _section(data_quality or (
            {"ranked": int(len(ranking)), "blocked": blocked} if ranking_ok else None), "no ranking run"),
        "universe": _section(None if universe is None else {
            "name": universe.name, "as_of": universe.as_of, "source": universe.source,
            "size": len(universe.symbols), "symbols": list(universe.symbols)}, "universe not loaded (BLOCKED)"),
        "top_quant_scores": _section(ranking.head(20) if ranking_ok else None, "no ranking"),
        "auction_leaders": _section(_leaders(ranking, "Auction"), "no auction data"),
        "tavan_dna_leaders": _section(_leaders(ranking, "Tavan-DNA"), "no Tavan-DNA data"),
        "volume_leaders": _section(_leaders(ranking, "Volume"), "no volume data"),
        "institutional_flow": _section(_leaders(ranking, "Institutional"), "no institutional data"),
        "kap_news": _section(events, "no KAP/news events"),
        "risk": _section(ranking[["Symbol", "Risk", "Entry", "Stop", "Target", "R/R"]].dropna(subset=["Risk"])
                         if ranking_ok else None, "no risk references"),
        "paper_trading": _section({"summary": paper_summary, "entries": paper_entries} if paper_summary else None,
                                  "paper ledger empty"),
        "auction": _section(auction, "no auction snapshots"),
        "daily_plan": _section(daily_plan, "daily plan not built"),
        "stress_test": _section(stress, "no portfolio / return history for stress test"),
        "overfit_check": _section(overfit, "no backtest trades for overfitting check"),
        "symbols": _plain(symbol_details or {}),
    }


def write_snapshot(snapshot: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return target


def blocked_snapshot(now: datetime, provider_statuses) -> dict[str, Any]:
    """What the dashboard shows when real providers are not configured."""
    return build_snapshot(now=now, provider_statuses=provider_statuses)
