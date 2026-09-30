"""Event study: what did each KAP/news event type historically do to the price?

Point-in-time rules
* Entry is the OPEN of the first bar strictly AFTER ``published_at``.
* The h-bar outcome is only known at the close of bar ``entry + h - 1``; with
  ``as_of`` set, outcomes that are not yet realized are NaN and excluded, so
  a table built "as of" a date never uses the future.
* "Priced in" compares the pre-event run-up (bars <= published_at) with the
  post-event reaction.
Sample sizes below ``min_events`` are reported as INSUFFICIENT_SAMPLE.
"""
from __future__ import annotations

from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd

from .event_features import NormalizedEvent

DEFAULT_HORIZONS = (1, 5, 20)


def _bars_by_symbol(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    data = bars.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    return {s: g.sort_values("timestamp").reset_index(drop=True) for s, g in data.groupby("symbol")}


def event_study(
    events: Iterable[NormalizedEvent],
    bars: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    runup_bars: int = 5,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    by_symbol = _bars_by_symbol(bars)
    as_of_utc = None if as_of is None else pd.Timestamp(as_of).tz_convert("UTC") if pd.Timestamp(as_of).tzinfo \
        else pd.Timestamp(as_of, tz="UTC")
    rows = []
    for ev in events:
        g = by_symbol.get(ev.symbol)
        if g is None or g.empty:
            continue
        published = pd.Timestamp(ev.published_at)
        if as_of_utc is not None and published > as_of_utc:
            continue  # the event itself is not known yet
        ts = g["timestamp"]
        entry_pos = int(ts.searchsorted(published, side="right"))  # first bar strictly after
        if entry_pos >= len(g):
            continue
        prior_pos = entry_pos - 1  # last bar at/before publication
        runup = None
        if prior_pos - runup_bars >= 0:
            runup = g.loc[prior_pos, "close"] / g.loc[prior_pos - runup_bars, "close"] - 1
        entry = g.loc[entry_pos, "open"]
        row = {"symbol": ev.symbol, "event_id": ev.event_id, "event_type": ev.event_type,
               "published_at": published, "entry_time": g.loc[entry_pos, "timestamp"], "runup": runup}
        for h in horizons:
            exit_pos = entry_pos + h - 1
            value = np.nan
            if exit_pos < len(g):
                known_at = g.loc[exit_pos, "timestamp"]
                if as_of_utc is None or known_at <= as_of_utc:
                    value = g.loc[exit_pos, "close"] / entry - 1
            row[f"ret_{h}"] = value
        rows.append(row)
    columns = ["symbol", "event_id", "event_type", "published_at", "entry_time", "runup",
               *[f"ret_{h}" for h in horizons]]
    return pd.DataFrame(rows, columns=columns)


def type_impact_table(study: pd.DataFrame, *, horizons: tuple[int, ...] = DEFAULT_HORIZONS,
                      min_events: int = 20) -> pd.DataFrame:
    """Per event type: n, mean/median return, hit rate, t-stat for each horizon."""
    out = []
    for event_type, g in study.groupby("event_type"):
        row: dict[str, object] = {"event_type": event_type, "events": int(len(g))}
        enough = True
        for h in horizons:
            r = g[f"ret_{h}"].dropna()
            n = len(r)
            row[f"n_{h}"] = n
            row[f"mean_{h}"] = float(r.mean()) if n else None
            row[f"median_{h}"] = float(r.median()) if n else None
            row[f"hit_{h}"] = float((r > 0).mean()) if n else None
            std = r.std(ddof=1) if n > 1 else 0.0
            row[f"t_{h}"] = float(r.mean() / (std / sqrt(n))) if n > 1 and std > 0 else None
            enough = enough and n >= min_events
        row["status"] = "OK" if enough else "INSUFFICIENT_SAMPLE"
        row["mean_runup"] = float(g["runup"].dropna().mean()) if g["runup"].notna().any() else None
        out.append(row)
    return pd.DataFrame(out).sort_values("events", ascending=False).reset_index(drop=True) if out \
        else pd.DataFrame(columns=["event_type", "events", "status"])


def priced_in_ratio(runup: float | None, reaction: float | None) -> str:
    """Has the move already happened? Compares pre-event run-up with post-event reaction."""
    if runup is None or reaction is None:
        return "UNKNOWN"
    if runup > 0.05 and reaction < runup * 0.25:
        return "LIKELY_PRICED_IN"
    if runup < 0.02 and reaction > 0.03:
        return "MARKET_SURPRISED"
    return "PARTIALLY_PRICED"


def rank_events_by_impact(events: Iterable[NormalizedEvent], table: pd.DataFrame, *, horizon: int = 5) -> list[dict]:
    """Order current events by the historical mean |return| of their type (most material first)."""
    lookup = {r["event_type"]: r for r in table.to_dict("records")}
    ranked = []
    for ev in events:
        stat = lookup.get(ev.event_type)
        usable = stat is not None and stat.get("status") == "OK" and stat.get(f"mean_{horizon}") is not None
        ranked.append({
            "symbol": ev.symbol, "event_id": ev.event_id, "event_type": ev.event_type, "headline": ev.headline,
            "historical_mean": stat[f"mean_{horizon}"] if usable else None,
            "historical_hit_rate": stat[f"hit_{horizon}"] if usable else None,
            "materiality": abs(stat[f"mean_{horizon}"]) if usable else None,
            "note": "historical type impact" if usable else "no reliable history for this event type",
        })
    return sorted(ranked, key=lambda r: (r["materiality"] is None, -(r["materiality"] or 0.0)))
