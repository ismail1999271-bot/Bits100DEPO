# Bits100 platform status — feat/live-data-stack (2026-09-30)

| Phase | Module(s) | State |
|---|---|---|
| 1 CI green | `.github/workflows/test.yml`, `scripts/check_no_order_routing.py` | ✅ compile + pytest + ruff + guard |
| 2 Quant Score + Tavan-DNA E2E | `quant_score.py`, `e2e_backtest.py`, `walk_forward.py` | ✅ coverage/confidence, purged date splits |
| 3 Tavan walk-forward | `tavan_validation.py` | ✅ per-fold metrics, untouched holdout |
| 4 Auction 09:40→09:55 | `auction_features.py` | ✅ snapshot/transition/trajectory features |
| 5 Single BIST100+ universe | `universe.py`, `daily_ranker.py` | ✅ point-in-time membership |
| 6 Provider contracts | `market_contracts.py`, `provider_status.py` | ✅ code · ⛔ credentials missing |
| 7 Level-2 / L2+ | `level2_features.py` | ✅ code · ⛔ vendor feed missing |
| 8 KAP/News/Fund/Broker/Inst. | `event_features.py` | ✅ code · ⛔ feeds missing |
| 8b Technical / volume | `technical.py` | ✅ 5M/15M/1H/4H/1D, IC usefulness |
| 9 Paper trading | `paper_ledger.py` | ✅ costs, slippage, liquidity, JSONL |
| 10 Dashboard | `dashboard_data.py`, `dashboard/app.py` | ✅ Streamlit, fail-closed snapshot |
| 11 Telegram | `telegram_notify.py` | ✅ outbound only |
| 12 Hardening | `research_run.py`, `fail_closed.py`, `AGENTS.md` | ✅ orchestrator + gate |
| Backtest / experiments / AI | `portfolio_backtest.py`, `experiments.py`, `ai_research.py` | ✅ |

## BLOCKED until real providers are configured
`BIST_MARKET_DATA_URL(+_TOKEN)`, `BIST_UNIVERSE_URL` or `BIST_SYMBOLS`, `BIST100_MEMBERSHIP_URL`,
`BIST_TICK_DATA_URL`, `BIST_LEVEL1_URL`, `BIST_LEVEL2_URL`, `BIST_LEVEL2PLUS_URL`,
`BIST_ORDER_EVENTS_URL`, `KAP_API_URL`, `NEWS_API_URL`, `FUND_FLOW_API_URL`, `BROKER_DATA_URL`,
`INSTITUTIONAL_DATA_URL` (each with `_TOKEN`), `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`.

No backtest in this repository has been run on real BIST data yet, so **no performance claim
exists**. Run `scripts/train_tavan_dna.py` / the portfolio backtest with `provenance=REAL` on a
licensed historical export to produce evidence.

## Run
```
pip install -e '.[dev,dashboard]'
python scripts/build_dashboard_snapshot.py && streamlit run dashboard/app.py
python -c "from bist_hunter.research_run import run_research; print(run_research().status)"
python scripts/ollama_task.py "<task>" --files <paths>   # local Ollama proposes a patch
```
