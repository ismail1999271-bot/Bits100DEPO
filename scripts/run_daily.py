"""Daily research run: real providers -> ranking -> charts -> snapshot -> plan -> (optional) Telegram.

Fail-closed: without configured providers the snapshot shows BLOCKED and no ranking is produced.
No order is ever sent. Telegram is outbound-only and only used when its env vars are set.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime

from bist_hunter.daily_plan import build_daily_plan, format_plan
from bist_hunter.dashboard_data import write_snapshot
from bist_hunter.research_run import run_research
from bist_hunter.telegram_notify import (
    SignalChangeTracker, change_report, daily_plan_message, notify, signal_report, system_status,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/dashboard/snapshot.json")
    parser.add_argument("--charts", default="artifacts/dashboard/charts")
    parser.add_argument("--state", default="artifacts/state/signal_status.json")
    parser.add_argument("--telegram", action="store_true", help="send outbound Telegram messages")
    args = parser.parse_args()
    now = datetime.now(UTC)
    run = run_research(now=now, chart_dir=args.charts)
    plan = build_daily_plan(now, run.ranking)
    snapshot = dict(run.snapshot)
    from bist_hunter.dashboard_data import _section

    snapshot["daily_plan"] = _section(plan, "daily plan not built")
    path = write_snapshot(snapshot, args.output)
    print(f"status={run.status} reason={run.reason} snapshot={path}")
    print(format_plan(plan))
    if args.telegram:
        notify(system_status(run.provider_statuses))
        notify(daily_plan_message(format_plan(plan)))
        if run.ranking is not None:
            tracker = SignalChangeTracker(args.state)
            changes = tracker.changes(run.ranking)
            tracker.save()
            report = change_report(changes)
            if report:
                notify(report)
                notify(signal_report(run.ranking))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
