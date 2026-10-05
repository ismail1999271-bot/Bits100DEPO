from datetime import datetime, timezone

import pandas as pd

from bist_hunter.daily_plan import build_daily_plan, format_plan


def test_plan_without_data_is_missing_not_invented():
    now = datetime(2026, 9, 30, 7, 0, tzinfo=timezone.utc)  # 10:00 Istanbul, Wednesday
    plan = build_daily_plan(now)
    assert plan.calendar["status"] == "MISSING"
    assert plan.watchlist == ()
    assert any(i.code == "AUCTION_FINAL" and i.done for i in plan.items)
    assert not next(i for i in plan.items if i.code == "POST_REVIEW").done
    assert "VERİ YOK" in format_plan(plan)


def test_plan_uses_ranking_and_calendar():
    now = datetime(2026, 9, 30, 4, 0, tzinfo=timezone.utc)
    ranking = pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"], "Status": ["OK", "BLOCKED", "OK"]})
    plan = build_daily_plan(now, ranking, economic_calendar=[{"date": "2026-09-30", "name": "PPK"},
                                                              {"date": "2026-10-01", "name": "x"}])
    assert plan.watchlist == ("AAA", "CCC") and plan.blocked == ("BBB",)
    assert len(plan.calendar["events"]) == 1


def test_snapshot_and_telegram_include_plan():
    from bist_hunter.dashboard_data import build_snapshot
    from bist_hunter.telegram_notify import daily_plan_message

    now = datetime(2026, 9, 30, 7, 0, tzinfo=timezone.utc)
    plan = build_daily_plan(now)
    snap = build_snapshot(now=now, daily_plan=plan)
    assert snap["daily_plan"]["status"] == "OK"
    assert snap["stress_test"]["status"] == "MISSING" and snap["overfit_check"]["status"] == "MISSING"
    assert "[DAILY_REPORT]" in daily_plan_message(format_plan(plan))
