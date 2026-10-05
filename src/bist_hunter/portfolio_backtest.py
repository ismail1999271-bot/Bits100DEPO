"""Realistic historical portfolio backtest: Train / Validation / Holdout.

Execution realism
* Decision at the close of day t using data <= t; fill at the OPEN of t+1;
  exit at the CLOSE of t+hold (no same-bar look-ahead).
* No fill when t+1 opens at the upper price limit (tavan queue is not
  fillable in practice) - conservative.
* Costs: commission + half spread + slippage on both legs.
* Liquidity: position value capped by participation of the prior 20-day
  average traded value; below ``min_adv_try`` the name is skipped.
* Survivorship: optional point-in-time UniverseHistory filters every day.

Evidence policy
* ``provenance`` must be "REAL" for results to be flagged as performance
  evidence. Anything else (e.g. SYNTHETIC test data) is labelled
  ``NOT_PERFORMANCE_EVIDENCE``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import pandas as pd

from .walk_forward import purged_date_split

EVIDENCE = "PERFORMANCE_EVIDENCE"
NOT_EVIDENCE = "NOT_PERFORMANCE_EVIDENCE"


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    top_k: int = 5
    score_threshold: float = 70.0
    hold_days: int = 1
    capital_try: float = 1_000_000.0
    commission_bps: float = 4.0
    spread_bps: float = 10.0
    slippage_bps: float = 5.0
    max_participation: float = 0.02
    min_adv_try: float = 5_000_000.0
    limit_pct: float = 0.10


@dataclass(frozen=True, slots=True)
class SegmentResult:
    segment: str
    period: tuple[str, str]
    total_return: float
    cagr: float | None
    win_rate: float
    expectancy: float
    profit_factor: float | None
    max_drawdown: float
    sharpe: float | None
    sortino: float | None
    trade_count: int
    turnover: float
    costs_try: float
    slippage_try: float
    skipped_limit_up: int
    skipped_liquidity: int


@dataclass(frozen=True, slots=True)
class BacktestResult:
    provenance: str
    evidence: str
    segments: tuple[SegmentResult, ...]
    trades: pd.DataFrame = field(repr=False)


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    g = data.groupby("symbol", group_keys=False)
    data["prev_close"] = g["close"].shift(1)
    data["next_open"] = g["open"].shift(-1)
    data["value"] = data["close"] * data["volume"]
    data["adv_20"] = g["value"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    return data


def _exit_close(data: pd.DataFrame, hold: int) -> pd.Series:
    return data.groupby("symbol")["close"].shift(-hold)


def _segment_metrics(name, trades: pd.DataFrame, dates, capital: float) -> SegmentResult:
    period = (str(min(dates)) if len(dates) else "", str(max(dates)) if len(dates) else "")
    if trades.empty:
        return SegmentResult(name, period, 0.0, None, 0.0, 0.0, None, 0.0, None, None, 0, 0.0, 0.0, 0.0,
                             0, 0)
    filled = trades[trades["status"] == "FILLED"]
    daily = filled.groupby("date")["pnl_try"].sum().reindex(sorted(set(dates)), fill_value=0.0)
    equity = capital + daily.cumsum()
    returns = daily / (capital + daily.cumsum().shift(1).fillna(0.0))
    peak = equity.cummax()
    max_dd = float(((peak - equity) / peak).max()) if len(equity) else 0.0
    total = float(equity.iloc[-1] / capital - 1) if len(equity) else 0.0
    years = max(1e-9, len(daily) / 252)
    cagr = float((1 + total) ** (1 / years) - 1) if len(daily) >= 20 and total > -1 else None
    std = returns.std(ddof=1)
    downside = np.sqrt((returns.clip(upper=0) ** 2).mean())
    sharpe = float(returns.mean() / std * sqrt(252)) if std and std > 0 else None
    sortino = float(returns.mean() / downside * sqrt(252)) if downside and downside > 0 else None
    wins = filled[filled["net_return"] > 0]
    losses = filled[filled["net_return"] < 0]
    pf = float(wins["pnl_try"].sum() / -losses["pnl_try"].sum()) if len(losses) and losses["pnl_try"].sum() else None
    return SegmentResult(
        name, period, round(total, 6), None if cagr is None else round(cagr, 6),
        round(float((filled["net_return"] > 0).mean()), 6) if len(filled) else 0.0,
        round(float(filled["net_return"].mean()), 8) if len(filled) else 0.0,
        None if pf is None else round(pf, 4), round(max_dd, 6),
        None if sharpe is None else round(sharpe, 4), None if sortino is None else round(sortino, 4),
        int(len(filled)), round(float(2 * filled["position_try"].sum() / capital / max(1, len(daily))), 6),
        round(float(filled["costs_try"].sum()), 2), round(float(filled["slippage_try"].sum()), 2),
        int((trades["status"] == "SKIP_LIMIT_UP").sum()), int((trades["status"] == "SKIP_LIQUIDITY").sum()),
    )


def run_portfolio_backtest(
    frame: pd.DataFrame,
    scores: pd.Series,
    *,
    provenance: str,
    config: ExecutionConfig = ExecutionConfig(),
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    universe_history=None,
) -> BacktestResult:
    """``scores`` is indexed like ``frame`` and must be computed point-in-time (score at close t)."""
    if not scores.index.equals(frame.index):
        raise ValueError("scores must share the frame index")
    data = _prepare(frame.assign(_score=scores))
    if universe_history is not None:
        from .universe import filter_point_in_time

        data = filter_point_in_time(data, universe_history)
    data["exit_close"] = _exit_close(data, config.hold_days)
    data["date"] = data["timestamp"].dt.date
    split = purged_date_split(data["timestamp"], train_fraction=train_fraction,
                              validation_fraction=validation_fraction, embargo_periods=config.hold_days)
    seg_of = {}
    for name, dates in (("train", split.train_dates), ("validation", split.validation_dates),
                        ("holdout", split.holdout_dates)):
        for d in dates:
            seg_of[pd.Timestamp(d).date()] = name

    per_trade_value = config.capital_try / config.top_k
    rows = []
    leg_bps = config.commission_bps + config.spread_bps / 2 + config.slippage_bps
    for date, day in data.dropna(subset=["_score"]).groupby("date"):
        segment = seg_of.get(date)
        if segment is None:
            continue
        picks = day[day["_score"] >= config.score_threshold].nlargest(config.top_k, "_score")
        for _, row in picks.iterrows():
            base = {"date": date, "segment": segment, "symbol": row["symbol"], "score": row["_score"]}
            if pd.isna(row["next_open"]) or pd.isna(row["exit_close"]):
                continue
            if row["next_open"] >= row["close"] * (1 + config.limit_pct) * 0.999:
                rows.append({**base, "status": "SKIP_LIMIT_UP"})
                continue
            adv = row["adv_20"]
            if pd.isna(adv) or adv < config.min_adv_try:
                rows.append({**base, "status": "SKIP_LIQUIDITY"})
                continue
            position = min(per_trade_value, adv * config.max_participation)
            gross = row["exit_close"] / row["next_open"] - 1
            costs = position * 2 * config.commission_bps / 10_000
            slippage = position * 2 * (config.spread_bps / 2 + config.slippage_bps) / 10_000
            net = gross - 2 * leg_bps / 10_000
            rows.append({**base, "status": "FILLED", "entry": row["next_open"], "exit": row["exit_close"],
                         "position_try": position, "gross_return": gross, "net_return": net,
                         "pnl_try": position * net, "costs_try": costs, "slippage_try": slippage})
    trades = pd.DataFrame(rows, columns=["date", "segment", "symbol", "score", "status", "entry", "exit",
                                         "position_try", "gross_return", "net_return", "pnl_try",
                                         "costs_try", "slippage_try"])
    segments = tuple(
        _segment_metrics(name, trades[trades["segment"] == name], [d for d, s in seg_of.items() if s == name],
                         config.capital_try)
        for name in ("train", "validation", "holdout")
    )
    provenance = provenance.upper()
    return BacktestResult(provenance, EVIDENCE if provenance == "REAL" else NOT_EVIDENCE, segments, trades)


def assert_point_in_time(score_fn, frame: pd.DataFrame, *, checkpoints: int = 3) -> None:
    """Recompute scores on truncated history; any difference proves look-ahead."""
    full = score_fn(frame)
    dates = sorted(pd.to_datetime(frame["timestamp"]).unique())
    for cut in np.linspace(len(dates) // 2, len(dates) - 2, checkpoints, dtype=int):
        cutoff = dates[cut]
        mask = pd.to_datetime(frame["timestamp"]) <= cutoff
        partial = score_fn(frame[mask])
        at_cut = pd.to_datetime(frame.loc[mask, "timestamp"]) == cutoff
        left = full[mask][at_cut].round(9)
        right = partial[at_cut].round(9)
        if not left.fillna(-1).equals(right.fillna(-1)):
            raise AssertionError(f"look-ahead detected at {cutoff}")
