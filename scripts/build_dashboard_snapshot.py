"""Build the dashboard snapshot from configured REAL providers (fail-closed).

Without providers this writes a BLOCKED snapshot that shows provider status
only - it never fills the dashboard with example numbers.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime

from bist_hunter.adapters import ProviderError
from bist_hunter.dashboard_data import build_snapshot, write_snapshot
from bist_hunter.provider_status import signals_allowed, status_board
from bist_hunter.universe import load_bist100_plus_universe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/dashboard/snapshot.json")
    args = parser.parse_args()
    now = datetime.now(UTC)
    statuses = status_board()
    try:
        universe = load_bist100_plus_universe(now.date())
    except ProviderError:
        universe = None
    snapshot = build_snapshot(now=now, universe=universe, provider_statuses=statuses)
    snapshot["signals_allowed"] = signals_allowed(statuses)
    path = write_snapshot(snapshot, args.output)
    print(f"snapshot -> {path} (signals_allowed={snapshot['signals_allowed']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
