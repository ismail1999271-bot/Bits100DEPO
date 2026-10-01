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

## Phase 13 – Extended research modules (overnight build)

All modules are research-only, fail-closed, and return MISSING/BLOCKED instead of estimating absent data.

| Module | Purpose | Data needed (else MISSING) |
|---|---|---|
| `tavan_risk.py` | VBTS/halt/IPO lock-up hard blocks, limit-up streak, lock strength, manipulation risk; wired into ranking (BLOCKED ranks last) | VBTS, trading state, L2 queue, free float (provider) |
| `technical.py` + `scenarios.py` | Weekly timeframe, bull/base/bear scenarios with invalidation levels | Real OHLCV |
| `event_study.py` | Historical price impact per event type, run-up / "priced in" ratio, INSUFFICIENT_SAMPLE gate | Real KAP/news history + OHLCV |
| `trade_review.py` | FIFO round trips; CHASE/EARLY_EXIT/OVERSIZED/REVENGE/STOP/OVERTRADING flags; process vs outcome | Paper ledger or user trade log |
| `stress_test.py` | Concentration, sector exposure, correlation, beta-based XU100 shocks | Return history + XU100 |
| `overfit.py` | Deflated Sharpe, regime performance, parameter sensitivity, verdict | Backtest trades (REAL provenance) |
| `daily_plan.py` | 06:00 → 18:10 routine, watchlist, calendar (MISSING without provider) | Ranking, economic calendar |

Still open: analyst-consensus "expectation surprise" (no provider), dashboard/Telegram panels for the new modules, real provider credentials.

## Phase 14 – Competitor-informed additions

| Module | Purpose | Notes |
|---|---|---|
| `borsapy_adapter.py` | Optional OHLCV backend (`BIST_DATA_BACKEND=borsapy`) | ~15 dk gecikmeli, kişisel/eğitim lisansı; canlı kullanım için ayrı BIST veri lisansı. Bulut ortamında erişilemediği için yalnız sahte Ticker ile test edildi; gerçek çağrı kullanıcının makinesinde denenmeli. Oturum açık iken bugünün yarım mumu düşürülür. |
| `news_classifier.py` | İki aşamalı haber/KAP sınıflandırma + piyasa bypass | LLM isteğe bağlı (`llm(prompt)->str`), yalnızca katı JSON şeması geçerse kabul; aksi halde kural tabanlı. Bypass aktifse tüm sıralama BLOCKED (MARKET_BYPASS). |
| `scan_presets.py` | 7 hazır tarama (hacim, kırılım, trend…) | Yalnız aday havuzunu daraltır; sinyal/skor üretmez. |
| `telegram_notify.SignalChangeTracker` | Yalnızca durum değişince bildirim | Bozuk state dosyası çökertmez. |
| `chart_analysis.py` | Mum grafiği, EMA/SMA, Bollinger, destek/direnç, formasyon, RSI/MACD | Pivotlar sağdan `confirm` bar ile doğrulanır (sızıntı yok); formasyonlar tahmin değildir. Dashboard "Hisse detayı" içinde. |
| `scripts/run_daily.py`, `Dockerfile`, `docker-compose.yml`, `daily-research.yml` | Günlük çalıştırma / konteyner | Sağlayıcı yoksa BLOCKED snapshot üretir. |

## Phase 15 – Canlı BIST takip ekranı (`dashboard/live.py`)

`streamlit run dashboard/live.py` (Docker: `live` servisi, port 8502). Tüm listelenen BIST hisseleri (evren: `BIST_SYMBOLS`/`BIST_UNIVERSE_URL`, `BIST_DATA_BACKEND=borsapy` ise `borsapy.companies()`).

* `live_board.py`: tavan/taban mesafesi, gün içi konum, rel. hacim, RSI/ATR/trend/kırılım, STALE işareti, uyarı üretimi.
* `live_feed.py`: fail-closed yükleyici. Kaynak yoksa BLOCKED; örnek veri gösterilmez.
* Sekmeler: Piyasa Haritası, Tüm Hisseler (filtre/arama), Hareketliler, Tavan/Taban, Hisse Analizi (grafik, destek/direnç, formasyon), Uyarılar. Otomatik yenileme 30–300 sn.
* Kaynak notu: BorsaPy ~15 dk gecikmeli ve kişisel/eğitim lisanslıdır; gerçek zamanlı için lisanslı Level-1 sağlayıcı gerekir (şu an bağlı değil).
* Doğrulama: birim testleri + Streamlit AppTest + tarayıcı ekran görüntüsü, hepsi **sentetik test verisiyle**. Gerçek BorsaPy çağrıları bulut ortamında erişilemediği için denenmedi.
