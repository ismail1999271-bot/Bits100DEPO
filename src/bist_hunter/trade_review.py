"""Post-trade review of the paper ledger: behavioural patterns and process score.

Round trips are built FIFO per symbol from ledger BUY/SELL entries. Patterns:
CHASE (bought after a big run-up), EARLY_EXIT (closed far short of plan),
OVERSIZED, REVENGE_TRADE (re-entry soon after a loss with a bigger size),
STOP_VIOLATION (held past the planned stop) and OVERTRADING.
Decision quality is judged on PROCESS (violations) separately from OUTCOME
(PnL), so a reasonable trade that lost money is not treated as a mistake.
Output is research feedback, never a trading instruction.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping

from .paper_ledger import LedgerEntry


@dataclass(frozen=True, slots=True)
class TradePlan:
    stop: float | None = None
    target: float | None = None
    prior_return: float | None = None  # run-up before entry (e.g. 5-day return)


@dataclass(frozen=True, slots=True)
class RoundTrip:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    fees: float
    notional: float
    reason_in: str
    reason_out: str

    @property
    def return_pct(self) -> float:
        return self.exit_price / self.entry_price - 1


@dataclass(frozen=True, slots=True)
class TradeVerdict:
    trip: RoundTrip
    flags: tuple[str, ...]
    outcome: str  # WIN / LOSS
    process: str  # CLEAN / FLAWED
    classification: str


@dataclass(frozen=True, slots=True)
class ReviewReport:
    trades: int
    verdicts: tuple[TradeVerdict, ...]
    pattern_counts: dict[str, int]
    process_score: float | None  # 0..100, share of clean trades
    good_process_bad_outcome: int
    bad_process_good_outcome: int
    rules: tuple[str, ...] = field(default_factory=tuple)
    status: str = "OK"


RULE_TEXT = {
    "CHASE": "Son 5 günde %{chase:.0f}'den fazla yükselmiş hisseye giriş yapma; geri çekilmeyi bekle.",
    "EARLY_EXIT": "Hedefin yarısına ulaşmadan kâr realize etme; çıkış nedeni yalnızca stop ya da hedef olsun.",
    "OVERSIZED": "Tek pozisyon sermayenin %{size:.0f}'ini geçmesin; boyutu girişten önce hesapla.",
    "REVENGE_TRADE": "Zararlı çıkıştan sonra en az {minutes} dakika yeni işlem açma; boyutu büyütme.",
    "STOP_VIOLATION": "Stop seviyesini önceden yaz ve ihlal edildiğinde ertesi seansı beklemeden çık.",
    "OVERTRADING": "Günde en fazla {daily} işlem yap; fazlası sinyal kalitesini düşürür.",
}
DEFAULT_RULES = (
    "Her işlemden önce giriş gerekçesi, stop ve hedefi yaz.",
    "Haftalık olarak işlem günlüğünü bu raporla gözden geçir.",
    "Kayıp serisinde (3 işlem) pozisyon boyutunu yarıya indir.",
    "Sadece Quant Score eşiğini ve risk filtresini geçen adaylara bak.",
    "Sonuç yerine süreç skoruna göre değerlendir: iyi süreç + zarar = hata değil.",
)


def build_round_trips(entries: Iterable[LedgerEntry]) -> list[RoundTrip]:
    lots: dict[str, deque] = defaultdict(deque)  # symbol -> deque[[qty, entry_price, entry_fee, ts, reason]]
    trips: list[RoundTrip] = []
    for e in entries:
        ts = datetime.fromisoformat(e.timestamp)
        if e.action == "BUY":
            lots[e.symbol].append([e.quantity, e.fill_price, e.fees, ts, e.reason])
        elif e.action == "SELL":
            remaining = e.quantity
            while remaining > 1e-12 and lots[e.symbol]:
                lot = lots[e.symbol][0]
                take = min(remaining, lot[0])
                share = take / lot[0]
                entry_fee = lot[2] * share
                exit_fee = e.fees * take / e.quantity
                pnl = (e.fill_price - lot[1]) * take - entry_fee - exit_fee
                trips.append(RoundTrip(e.symbol, lot[3], ts, take, lot[1], e.fill_price, pnl,
                                       entry_fee + exit_fee, lot[1] * take, lot[4], e.reason))
                lot[0] -= take
                lot[2] -= entry_fee
                remaining -= take
                if lot[0] <= 1e-12:
                    lots[e.symbol].popleft()
    return sorted(trips, key=lambda t: t.entry_time)


def review_trades(
    entries: Iterable[LedgerEntry],
    *,
    initial_equity: float,
    plans: Mapping[tuple[str, str], TradePlan] | None = None,
    max_position_pct: float = 0.10,
    chase_threshold: float = 0.06,
    revenge_minutes: int = 60,
    max_daily_trades: int = 5,
    early_exit_fraction: float = 0.5,
) -> ReviewReport:
    """``plans`` maps (symbol, entry_time_iso) -> TradePlan for plan-dependent checks."""
    trips = build_round_trips(entries)
    if not trips:
        return ReviewReport(0, (), {}, None, 0, 0, DEFAULT_RULES, "NO_CLOSED_TRADES")
    plans = plans or {}
    per_day = Counter(t.entry_time.date() for t in trips)
    verdicts: list[TradeVerdict] = []
    last_loss_exit: datetime | None = None
    last_loss_size: float | None = None
    for trip in trips:
        flags: list[str] = []
        plan = plans.get((trip.symbol, trip.entry_time.isoformat()), TradePlan())
        if plan.prior_return is not None and plan.prior_return >= chase_threshold:
            flags.append("CHASE")
        if plan.target is not None and plan.target > trip.entry_price and trip.pnl > 0 \
                and trip.reason_out.lower() not in ("stop", "target") \
                and (trip.exit_price - trip.entry_price) < early_exit_fraction * (plan.target - trip.entry_price):
            flags.append("EARLY_EXIT")
        if trip.notional > initial_equity * max_position_pct * 1.0:
            flags.append("OVERSIZED")
        if last_loss_exit is not None and 0 <= (trip.entry_time - last_loss_exit).total_seconds() <= revenge_minutes * 60 \
                and last_loss_size is not None and trip.notional > last_loss_size:
            flags.append("REVENGE_TRADE")
        if plan.stop is not None and trip.exit_price < plan.stop * 0.995 and trip.reason_out.lower() != "stop":
            flags.append("STOP_VIOLATION")
        if per_day[trip.entry_time.date()] > max_daily_trades:
            flags.append("OVERTRADING")
        if trip.pnl < 0:
            last_loss_exit, last_loss_size = trip.exit_time, trip.notional
        outcome = "WIN" if trip.pnl > 0 else "LOSS"
        process = "FLAWED" if flags else "CLEAN"
        classification = (
            "GOOD_PROCESS_BAD_OUTCOME" if process == "CLEAN" and outcome == "LOSS"
            else "BAD_PROCESS_LUCKY_OUTCOME" if process == "FLAWED" and outcome == "WIN"
            else "BAD_PROCESS_BAD_OUTCOME" if process == "FLAWED" else "GOOD_PROCESS_GOOD_OUTCOME")
        verdicts.append(TradeVerdict(trip, tuple(flags), outcome, process, classification))
    counts = Counter(f for v in verdicts for f in v.flags)
    clean = sum(v.process == "CLEAN" for v in verdicts)
    params = {"chase": chase_threshold * 100, "size": max_position_pct * 100, "minutes": revenge_minutes,
              "daily": max_daily_trades}
    rules = [RULE_TEXT[name].format(**params) for name, _ in counts.most_common(5)]
    for default in DEFAULT_RULES:
        if len(rules) >= 5:
            break
        rules.append(default)
    return ReviewReport(
        len(verdicts), tuple(verdicts), dict(counts), round(100.0 * clean / len(verdicts), 2),
        sum(v.classification == "GOOD_PROCESS_BAD_OUTCOME" for v in verdicts),
        sum(v.classification == "BAD_PROCESS_LUCKY_OUTCOME" for v in verdicts), tuple(rules[:5]),
    )
