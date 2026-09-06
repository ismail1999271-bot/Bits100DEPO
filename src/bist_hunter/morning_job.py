"""06:00 research job orchestration.

The job is safe by default: without a configured market-data endpoint it emits
an explicit NO_LIVE_DATA state instead of inventing a signal.
"""
from dataclasses import dataclass
import os
from datetime import datetime, timezone

from .adapters import BistMarketDataAdapter, HttpJsonProvider, ProviderError, parse_json_records
from .telegram_report import send_message


@dataclass(frozen=True, slots=True)
class MorningResult:
    status: str
    message: str


def build_morning_message(records: list[dict]) -> str:
    if not records:
        return "🔴 BUGÜN KALİTELİ TAVAN SİNYALİ YOK\nVeri kaynağından aday üretilemedi."
    return "🌅 BIST HUNTER — 06:00\n" + "\n".join(
        f"{i}. {r.get('symbol', '?')} | skor={r.get('score', '?')}" for i, r in enumerate(records[:20], 1)
    )


def run_morning_job() -> MorningResult:
    endpoint = os.getenv("BIST_MARKET_DATA_URL", "")
    if not endpoint:
        return MorningResult("NO_LIVE_DATA", "🔴 Canlı BIST veri sağlayıcısı yapılandırılmamış; sinyal üretilmedi.")
    try:
        adapter = BistMarketDataAdapter(HttpJsonProvider(endpoint))
        payload = adapter.bars("ALL", "latest", datetime.now(timezone.utc).date().isoformat())
        records = parse_json_records(payload)
        message = build_morning_message(records)
        sent = send_message(message)
        return MorningResult("SENT" if sent else "GENERATED", message)
    except ProviderError as exc:
        return MorningResult("PROVIDER_ERROR", f"🔴 Veri sağlayıcı hatası: {exc}")
