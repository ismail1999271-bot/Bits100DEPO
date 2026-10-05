"""Daily market routine / plan generator (research only, no orders).

Builds the day's checklist from the clock and from REAL inputs. Economic
calendar items are shown only when supplied by a provider; otherwise the
section says MISSING. Exchange holidays are not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

import pandas as pd

from .auction_features import TZ

ROUTINE: tuple[tuple[time, str, str], ...] = (
    (time(6, 0), "PRE_REPORT", "Gece haberleri, KAP, küresel piyasalar, ekonomik takvim özeti"),
    (time(9, 40), "AUCTION_OPEN", "Açılış seansı başladı: emir defteri/indikatif fiyat izle"),
    (time(9, 45), "AUCTION_CHECK_1", "Indikatif fiyat/hacim değişimini kaydet"),
    (time(9, 50), "AUCTION_CHECK_2", "Emir dengesizliği ve manipülasyon işaretlerini kontrol et"),
    (time(9, 55), "AUCTION_FINAL", "Eşleşme öncesi son durum; aday listesini güncelle"),
    (time(10, 0), "WATCHLIST", "Sürekli işlem: izleme listesi ve risk referansları"),
    (time(12, 0), "MIDDAY_ALERTS", "Gün içi uyarılar: hacim/fiyat sapmaları, KAP akışı"),
    (time(17, 50), "PRE_CLOSE", "Kapanış seansı öncesi pozisyon ve risk kontrolü"),
    (time(18, 10), "POST_REVIEW", "Gün sonu değerlendirme: sinyal-sonuç, işlem hatası analizi"),
)


@dataclass(frozen=True, slots=True)
class PlanItem:
    at: str  # HH:MM Istanbul
    code: str
    text: str
    done: bool


@dataclass(frozen=True, slots=True)
class DailyPlan:
    date: str
    phase_note: str
    items: tuple[PlanItem, ...]
    watchlist: tuple[str, ...]
    blocked: tuple[str, ...]
    calendar: dict
    warnings: tuple[str, ...]
    disclaimer: str = "Araştırma platformudur; yatırım tavsiyesi değildir. Otomatik emir yoktur."


def build_daily_plan(now: datetime, ranking: pd.DataFrame | None = None, *,
                     economic_calendar: list[dict] | None = None, top_n: int = 10) -> DailyPlan:
    local = now.astimezone(TZ)
    warnings: list[str] = []
    if local.weekday() >= 5:
        warnings.append("weekend_no_session")
    warnings.append("holiday_calendar_not_checked")
    items = tuple(PlanItem(t.strftime("%H:%M"), code, text, local.time() >= t) for t, code, text in ROUTINE)

    watch: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    if ranking is None or ranking.empty:
        warnings.append("no_ranking_watchlist_empty")
    else:
        ok = ranking[ranking["Status"] != "BLOCKED"]
        watch = tuple(ok.head(top_n)["Symbol"].tolist())
        blocked = tuple(ranking[ranking["Status"] == "BLOCKED"]["Symbol"].tolist())

    today = local.date().isoformat()
    if economic_calendar is None:
        calendar = {"status": "MISSING", "reason": "economic calendar provider not configured", "events": None}
    else:
        todays = [e for e in economic_calendar if str(e.get("date", ""))[:10] == today]
        calendar = {"status": "OK", "events": todays}
    return DailyPlan(today, local.strftime("%A %H:%M"), items, watch, blocked, calendar, tuple(warnings))


def format_plan(plan: DailyPlan) -> str:
    lines = [f"GÜNLÜK PLAN {plan.date} ({plan.phase_note})"]
    for i in plan.items:
        lines.append(f"{'[x]' if i.done else '[ ]'} {i.at} {i.text}")
    lines.append("İzleme listesi: " + (", ".join(plan.watchlist) if plan.watchlist else "YOK (sıralama yok)"))
    if plan.blocked:
        lines.append("Engelli: " + ", ".join(plan.blocked))
    if plan.calendar["status"] == "MISSING":
        lines.append("Ekonomik takvim: VERİ YOK")
    else:
        lines.extend(f"Takvim: {e}" for e in plan.calendar["events"]) if plan.calendar["events"] else \
            lines.append("Takvim: bugün kayıt yok")
    if plan.warnings:
        lines.append("Uyarılar: " + ", ".join(plan.warnings))
    lines.append(plan.disclaimer)
    return "\n".join(lines)
