# CI status

The live-data stack branch is maintained fail-closed.

## Current gate

- Python compile: must pass
- Pytest: must pass with zero failures
- Ruff: must pass
- Production guard: must pass
- Real market-data providers: require externally supplied credentials

A provider without credentials must produce a blocked readiness state rather than synthetic market data.

## Faz 1 (CI green) — 2026-09-30

- `Tests` workflow now also runs on pushes to `feat/**` and via `workflow_dispatch`.
- Ruff gate runs on the full repo (`ruff check .`), not only `src`.
- Production guard is `scripts/check_no_order_routing.py`: it fails the build if any
  source defines/calls exchange order routing (`create_order`, `place_order`, `cancel_order`, ...).
- `technology-scout.yml`: removed the invalid job-level `secrets` condition; the job now
  reports `BLOCKED_NO_AI_PROVIDER` when the key is absent.
- Paper broker test isolates limit logic from the spread/slippage model; a new test proves
  execution costs count against the position limit.
