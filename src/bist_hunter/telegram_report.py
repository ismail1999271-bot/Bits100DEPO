"""Minimal Telegram Bot API client; disabled unless explicitly configured."""
from __future__ import annotations
import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def send_message(text: str, token: str | None = None, chat_id: str | None = None) -> bool:
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False
    payload = urlencode({"chat_id": chat_id, "text": text}).encode()
    request = Request(f"https://api.telegram.org/bot{token}/sendMessage", data=payload, method="POST")
    with urlopen(request, timeout=15) as response:
        body = json.loads(response.read().decode("utf-8"))
    return bool(body.get("ok"))
