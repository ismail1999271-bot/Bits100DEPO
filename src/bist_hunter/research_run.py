"""End-to-end daily research run on REAL providers (fail-closed orchestrator).

Real data -> validation -> BIST100+ universe -> market data -> (Level-2) ->
(auction) -> technical/volume -> Tavan-DNA -> KAP/news/fund/broker/institutional
-> Quant Score -> ranking -> risk filter -> dashboard snapshot -> Telegram.

If a required provider is not CONNECTED the run returns status BLOCKED with
the provider board; no ranking is produced from partial or synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from .adapters import ProviderError
from .dashboard_data import build_snapshot
from .event_features import (
    broker_features, deduplicate, event_score, institutional_features, normalize_kap, normalize_news,
    quant_components,
)
from .fail_closed import GateInput
from .news_classifier import classify_events, market_bypass, news_score
from .provider_status import CONNECTED, ProviderClient, ProviderStatus, signals_allowed, status_board
from .research_ranking import SymbolResearch, rank_research
from .risk import RiskInputs
from .tavan_risk import TavanRiskInputs, assess_tavan_risk, closes_from_frame
from .tavan_model import FEATURES, TavanLogisticModel, build_tavan_dataset
from .technical import add_technical_features, technical_score
from .universe import load_bist100_plus_universe


@dataclass(frozen=True, slots=True)
class ResearchRun:
    status: str
    reason: str
    provider_statuses: tuple[ProviderStatus, ...]
    ranking: pd.DataFrame | None = None
    snapshot: dict = field(default_factory=dict)


def _bars_frame(bars) -> pd.DataFrame:
    return pd.DataFrame([{"symbol": b.symbol, "timestamp": b.timestamp, "open": b.open, "high": b.high,
                          "low": b.low, "close": b.close, "volume": b.volume} for b in bars])


def run_research(
    *,
    now: datetime | None = None,
    client: ProviderClient | None = None,
    history_days: int = 400,
    max_market_age_seconds: float = 26 * 3600,
) -> ResearchRun:
    now = now or datetime.now(timezone.utc)
    client = client or ProviderClient()
    board = {s.domain: s for s in status_board(client.env)}
    if not signals_allowed(tuple(board.values())):
        return ResearchRun("BLOCKED", "required provider not configured", tuple(board.values()),
                           snapshot=build_snapshot(now=now, provider_statuses=tuple(board.values())))
    try:
        universe = load_bist100_plus_universe(now.date())
    except ProviderError as exc:
        return ResearchRun("BLOCKED", f"universe: {exc}", tuple(board.values()),
                           snapshot=build_snapshot(now=now, provider_statuses=tuple(board.values())))

    params = {"symbols": ",".join(universe.symbols), "start": (now - timedelta(days=history_days)).date().isoformat(),
              "end": now.date().isoformat()}
    ohlcv_status, bars = client.fetch("ohlcv", params, now=now)
    board["ohlcv"] = ohlcv_status
    if ohlcv_status.status != CONNECTED:
        return ResearchRun("BLOCKED", f"ohlcv {ohlcv_status.status}: {ohlcv_status.reason}", tuple(board.values()),
                           snapshot=build_snapshot(now=now, universe=universe,
                                                   provider_statuses=tuple(board.values())))
    frame = _bars_frame(bars)

    events_params = {"symbols": params["symbols"], "start": (now - timedelta(days=3)).isoformat(),
                     "end": now.isoformat()}
    kap_status, kap_raw = client.fetch("kap", events_params, now=now)
    news_status, news_raw = client.fetch("news", events_params, now=now)
    broker_status, broker_rows = client.fetch("broker", events_params, now=now)
    inst_status, inst_rows = client.fetch("institutional", events_params, now=now)
    for s in (kap_status, news_status, broker_status, inst_status):
        board[s.domain] = s
    kap_events = deduplicate(normalize_kap([dict(e.payload) for e in kap_raw]))
    news_events = deduplicate(normalize_news([dict(e.payload) for e in news_raw]))
    news_classes = classify_events(news_events + kap_events, now)
    bypass = market_bypass(news_events, now)

    # Tavan-DNA: fit on labelled history strictly before today, score today's rows.
    labeled = build_tavan_dataset(frame)
    latest_day = frame["timestamp"].max()
    train = labeled[labeled["timestamp"] < latest_day - pd.Timedelta(days=1)]
    tech = add_technical_features(frame)
    latest = tech.sort_values("timestamp").groupby("symbol").tail(1).set_index("symbol")
    probabilities: dict[str, float] = {}
    if len(train) >= 30 and train["target"].nunique() == 2:
        model = TavanLogisticModel().fit(train)
        last_feats = _tavan_features_last_bar(frame)
        last_feats = last_feats.dropna(subset=list(FEATURES))
        if not last_feats.empty:
            for symbol, p in zip(last_feats["symbol"], model.predict_proba(last_feats)):
                probabilities[symbol] = float(p)

    items = []
    for symbol in universe.symbols:
        if symbol not in latest.index:
            continue
        row = latest.loc[symbol]
        comps: dict[str, float | None] = {"technical": technical_score(row)}
        if symbol in probabilities:
            comps["tavan_dna"] = probabilities[symbol] * 100
        comps.update(quant_components(
            kap=event_score(kap_events, symbol, now) if kap_events else None,
            news=_news_component(news_events, news_classes, symbol, now),
            broker=broker_features(broker_rows, symbol, now) if broker_rows else None,
            institutional=institutional_features(inst_rows, symbol, now) if inst_rows else None,
        ))
        adv = float(frame[frame["symbol"] == symbol].sort_values("timestamp").tail(20).eval("close * volume").mean())
        items.append(SymbolResearch(
            symbol,
            GateInput(symbol, now, row["timestamp"].to_pydatetime() if hasattr(row["timestamp"], "to_pydatetime")
                      else row["timestamp"], max_market_age_seconds, None,
                      provider_statuses={d: s.status for d, s in board.items()}),
            comps,
            relative_volume=None if pd.isna(row["relative_volume"]) else float(row["relative_volume"]),
            tavan_risk=assess_tavan_risk(TavanRiskInputs(symbol, now, daily_closes=closes_from_frame(frame, symbol))),
            risk=RiskInputs(last_price=float(row["close"]), atr=None if pd.isna(row["atr_14"]) else float(row["atr_14"]),
                            support=None if pd.isna(row["support_20"]) else float(row["support_20"]),
                            reference_close=float(row["close"]), avg_daily_value_try=adv),
        ))
    ranking = rank_research(universe, items, bypass=bypass)
    snapshot = build_snapshot(now=now, universe=universe, provider_statuses=tuple(board.values()), ranking=ranking)
    return ResearchRun("OK", "ranked", tuple(board.values()), ranking, snapshot)


def _tavan_features_last_bar(frame: pd.DataFrame) -> pd.DataFrame:
    """Tavan-DNA features for each symbol's last bar (label not needed for scoring)."""
    padded = []
    for symbol, g in frame.sort_values("timestamp").groupby("symbol"):
        # Append a sentinel next bar so build_tavan_dataset keeps the real last row; the
        # sentinel only feeds the (discarded) label, never a feature.
        last = g.iloc[-1]
        sentinel = last.copy()
        sentinel["timestamp"] = last["timestamp"] + pd.Timedelta(days=1)
        padded.append(pd.concat([g, sentinel.to_frame().T]))
    data = pd.concat(padded, ignore_index=True)
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = data[column].astype(float)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    built = build_tavan_dataset(data)
    last_ts = frame.groupby("symbol")["timestamp"].max()
    built = built[built.apply(lambda r: pd.Timestamp(r["timestamp"]) == pd.Timestamp(last_ts[r["symbol"]]), axis=1)]
    return built[["symbol", "timestamp", *FEATURES]]


def _news_component(news_events, news_classes, symbol, now):
    """Classifier-based news score when it has evidence, else the keyword baseline."""
    if not news_events:
        return None
    score = news_score(news_classes, symbol)
    if score is not None:
        from .event_features import EventScore, OK
        return EventScore(symbol, OK, score, sum(c.symbol == symbol for c in news_classes), None)
    return event_score(news_events, symbol, now)
