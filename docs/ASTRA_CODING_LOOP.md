# GPT-6 Astra Coding Loop

Bits100 uses GPT-6 Astra as a senior software engineer for repository completion, not as a trading authority.

## What Astra is asked to do

- Audit the full checked-out repository for incomplete production-critical software.
- Identify the highest-value gaps in data adapters, point-in-time integrity, ranking, Tavan-DNA training, walk-forward validation, paper trading, agent feedback, observability and CI/CD.
- Implement concrete fixes rather than returning a design-only answer.
- Add tests for behavioral changes.
- Preserve fail-closed behavior whenever real credentials, licensed data or external services are unavailable.

## Verification loop

1. Snapshot repository.
2. GPT-6 Astra proposes a unified diff.
3. Patch is checked and applied.
4. Compile gate runs.
5. Pytest runs.
6. Ruff runs.
7. If any gate fails, the pass is reverted and the failure output is fed back to Astra.
8. Up to three implementation/repair passes are allowed.
9. A successful pass is committed on an isolated `astra/*` branch and opened as a **draft PR**.

## Hard constraints

Astra must not:

- invent BIST prices, KAP disclosures, fund flows, broker targets or institutional holdings;
- bypass authentication or licensed data access;
- introduce live trading or order execution;
- remove risk gates to make tests pass;
- use future information in historical features/backtests;
- expose API keys or other secrets to source files or model output.

GPT output is a software-engineering input. Production promotion remains gated by deterministic tests, out-of-sample validation and the existing production-readiness controls.

## Required secret

GitHub Actions requires `OPENAI_API_KEY` as a repository Actions secret. The workflow pins the model through `OPENAI_MODEL=gpt-6-astra`.
