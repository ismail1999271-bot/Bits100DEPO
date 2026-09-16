# CI status

The live-data stack branch is maintained fail-closed.

## Current gate

- Python compile: must pass
- Pytest: must pass with zero failures
- Ruff: must pass
- Production guard: must pass
- Real market-data providers: require externally supplied credentials

A provider without credentials must produce a blocked readiness state rather than synthetic market data.
