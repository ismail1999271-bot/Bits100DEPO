"""Outbound-only Telegram notifications.

Allowed message kinds: notification, alert, daily report, signal (research)
report, system status and CI status. The bot never reads updates, has no
command handlers and cannot trigger orders - there is no inbound path.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from .telegram_report import send_message

KINDS = ("NOTIFICATION", "ALERT", "DAILY_REPORT", "SIGNAL_REPORT", "SYSTEM_STATUS", "CI_STATUS")
MAX_LEN = 4000
FOOTER = "ℹ️ Araştırma bildirimi; yatırım tavsiyesi değildir. Otomatik emir yoktur."
ICON = {"CONNECTED": "🟢", "STALE": "🟡", "INVALID": "🔴", "MISSING": "⚪", "BLOCKED": "⛔"}


def _chunks(text: str) -> list[str]:
    lines, chunks, current = text.splitlines(), [], ""
    for line in lines:
        if len(current) + len(line) + 1 > MAX_LEN and current:
            chunks.append(current)
            current = ""
        current += line[:MAX_LEN] + "\n"
    if current:
        chunks.append(current)
    return chunks


def format_message(kind: str, title: str, body: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"unsupported Telegram message kind: {kind}")
    return f"[{kind}] {title}\n\n{body.strip()}\n\n{FOOTER}"


def signal_report(ranking: pd.DataFrame, *, top: int = 10, universe_label: str = "") -> str:
    ok = ranking[ranking["Status"] == "SIGNAL"].head(top)
    lines = [f"Evren: {universe_label}" if universe_label else "Evren: BIST100+"]
    if ok.empty:
        lines.append("🔴 Kalite eşiğini geçen araştırma adayı yok.")
    for _, row in ok.iterrows():
        rr = "—" if pd.isna(row.get("R/R")) else row["R/R"]
        lines.append(f"{int(row['Rank'])}. {row['Symbol']} | QS {row['Quant Score']} | cov {row['Coverage']:.2f}"
                     f" | risk {row.get('Risk') or '—'} | R/R {rr}")
    blocked = int((ranking["Status"] == "BLOCKED").sum())
    lines.append(f"BLOCKED sembol: {blocked}")
    return format_message("SIGNAL_REPORT", "Günlük araştırma sıralaması", "\n".join(lines))


def system_status(statuses) -> str:
    body = "\n".join(f"{ICON.get(s.status, '')} {s.label}: {s.status}" for s in statuses)
    return format_message("SYSTEM_STATUS", "Veri sağlayıcı durumu", body)


def ci_status(branch: str, sha: str, passed: bool, details: str = "") -> str:
    head = f"{'✅ PASS' if passed else '❌ FAIL'} · {branch} @ {sha[:7]}"
    return format_message("CI_STATUS", "CI", head + ("\n" + details if details else ""))


def notify(text: str, *, sender: Callable[[str], bool] = send_message) -> bool:
    """Send (possibly chunked) text. Returns False when Telegram is not configured."""
    return all(sender(chunk) for chunk in _chunks(text))
