from __future__ import annotations

import hashlib
import re
import unicodedata

from bist_hunter.data.schema import NormalizedNews


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", title).casefold()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def news_fingerprint(item: NormalizedNews) -> str:
    key = f"{item.published_at.isoformat()}|{item.source}|{normalize_title(item.title)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def deduplicate(items: list[NormalizedNews]) -> list[NormalizedNews]:
    seen: set[str] = set()
    result: list[NormalizedNews] = []
    for item in sorted(items, key=lambda x: x.published_at):
        fp = news_fingerprint(item)
        if fp not in seen:
            seen.add(fp)
            result.append(item)
    return result
