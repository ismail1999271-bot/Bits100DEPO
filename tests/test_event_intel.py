from datetime import datetime

from bist_hunter.event_intel import SourceProfile, normalize_event_batch
from bist_hunter.news import EventType, NewsEvent


def test_event_quality_penalizes_hype_and_duplicates():
    event = NewsEvent(
        source="social",
        published_at=datetime(2026, 9, 6, 5),
        title="AAA tavan uçacak garanti",
        url="https://example/1",
        event_type=EventType.SOCIAL,
        symbols=("AAA",),
        sentiment=1.0,
        reliability=0.4,
    )
    result = normalize_event_batch([event, event], {"social": SourceProfile("social", 0.4)})
    assert len(result) == 1
    assert result[0].manipulation_risk > 0
    assert result[0].reliability < 0.5
