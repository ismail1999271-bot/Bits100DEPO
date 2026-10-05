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


def daily_plan_message(plan_text: str) -> str:
    return format_message("DAILY_REPORT", "Günlük plan", plan_text)


def system_status(statuses) -> str:
    body = "\n".join(f"{ICON.get(s.status, '')} {s.label}: {s.status}" for s in statuses)
    return format_message("SYSTEM_STATUS", "Veri sağlayıcı durumu", body)


def ci_status(branch: str, sha: str, passed: bool, details: str = "") -> str:
    head = f"{'✅ PASS' if passed else '❌ FAIL'} · {branch} @ {sha[:7]}"
    return format_message("CI_STATUS", "CI", head + ("\n" + details if details else ""))


def notify(text: str, *, sender: Callable[[str], bool] = send_message) -> bool:
    """Send (possibly chunked) text. Returns False when Telegram is not configured."""
    return all(sender(chunk) for chunk in _chunks(text))


class SignalChangeTracker:
    """Notify only when a symbol's status changes (no repeated identical alerts).

    State is a plain JSON file {symbol: status}. First sight of a symbol counts as a change.
    """

    def __init__(self, path: str | None = None) -> None:
        import json
        from pathlib import Path

        self._json = json
        self._path = Path(path) if path else None
        self.state: dict[str, str] = {}
        if self._path is not None and self._path.exists():
            try:
                self.state = {str(k): str(v) for k, v in json.loads(self._path.read_text("utf-8")).items()}
            except (ValueError, OSError):
                self.state = {}  # corrupt state -> treat everything as new, never crash

    def changes(self, ranking: pd.DataFrame) -> list[tuple[str, str | None, str]]:
        out = []
        for _, row in ranking.iterrows():
            symbol, status = str(row["Symbol"]), str(row["Status"])
            previous = self.state.get(symbol)
            if previous != status:
                out.append((symbol, previous, status))
            self.state[symbol] = status
        return out

    def save(self) -> None:
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(self._json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")


def change_report(changes: list[tuple[str, str | None, str]], *, only_to: tuple[str, ...] = ("SIGNAL", "BLOCKED")) -> str | None:
    lines = [f"{s}: {old or 'yeni'} → {new}" for s, old, new in changes if new in only_to]
    if not lines:
        return None
    return format_message("ALERT", "Durum değişikliği", "\n".join(lines[:40]))
