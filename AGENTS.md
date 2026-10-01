# AGENTS.md — rules for every human or AI contributor (Claude, Astra, Ollama, Codex)

Bits100 is a **BIST100+ research and signal platform**, not a trading bot.

## Non-negotiable
1. **No automatic orders.** Never add `create_order`, `place_order`, `cancel_order` or any
   exchange order routing. `scripts/check_no_order_routing.py` fails CI if you do.
   Telegram and AI never get order authority.
2. **No fake data in production.** Missing credentials/endpoints → `BLOCKED`/`MISSING`.
   Never fill missing values with 0, estimates or synthetic data. `synthetic.py` is for tests only.
3. **No future leakage.** Features at time *t* use data ≤ *t*. Splits are chronological on dates
   with an embargo. KAP/news are usable only after `published_at`.
4. **No survivorship bias.** Backtests use point-in-time `UniverseHistory`; one BIST100+ universe
   (`universe.load_bist100_plus_universe`) for every pipeline.
5. **Final holdout is never used for tuning.**
6. **Only `provenance=REAL` results are performance evidence.**
7. **CI must be green before moving on:** `python -m compileall -q src tests scripts dashboard`,
   `pytest -q`, `ruff check .`, `python scripts/check_no_order_routing.py`.

## AI output
AI may propose patterns, hypotheses, features, backtest designs, anomaly explanations and code
changes. Flow: AI Hypothesis → Quant Engine → Backtest → Validation → Research Result
(`ai_research.py`). AI output is never a trading signal. Local models: `scripts/ollama_task.py`
(proposes a patch; a human applies it and runs the CI gate).

## Pipeline map (src/bist_hunter)
Real data `provider_status.py`/`market_contracts.py` → validation `fail_closed.py` → universe
`universe.py` → market data → Level-2 `level2_features.py` → auction `auction_features.py`
(09:40→09:45→09:50→09:55) → technical/volume `technical.py` → Tavan-DNA `tavan_model.py`,
`tavan_validation.py` → KAP/news/fund/broker/institutional `event_features.py` → Quant Score
`quant_score.py` → ranking/risk `research_ranking.py`, `risk.py` → backtest `e2e_backtest.py`,
`portfolio_backtest.py` → paper `paper_ledger.py` → dashboard `dashboard_data.py`, `dashboard/app.py`
→ Telegram `telegram_notify.py`. Orchestrator: `research_run.py`. Experiments: `experiments.py`.

Extended research modules (Phase 13): `tavan_risk.py` (hard blocks, wired in `research_run.py`),
`scenarios.py`, `event_study.py`, `trade_review.py`, `stress_test.py`, `overfit.py`, `daily_plan.py`.
Each returns MISSING/BLOCKED rather than estimating absent data; none routes orders.
Phase 14: `borsapy_adapter.py`, `news_classifier.py`, `scan_presets.py`, `chart_analysis.py`, `scripts/run_daily.py`.
