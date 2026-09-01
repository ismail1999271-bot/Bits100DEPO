# Bits100DEPO — 20-Step Build Plan

1. Stabilize package structure and imports.
2. Make pytest fully green.
3. Make Ruff/lint fully green.
4. Add deterministic end-to-end synthetic pipeline test.
5. Harden OHLCV normalization and data-quality checks.
6. Implement leakage-safe Tavan DNA/event labeling.
7. Build daily opportunity feature engine.
8. Build Top-K ranking and confidence gates.
9. Add chronological train/test and walk-forward evaluation.
10. Add realistic commission, slippage and liquidity constraints.
11. Implement evaluation: precision, recall, hit-day rate, lead time, drawdown and net P&L.
12. Integrate a compliant/licensed BIST historical market-data provider through an adapter.
13. Integrate KAP disclosures with timestamps and symbol mapping.
14. Integrate reputable news feeds and deduplication/sentiment pipeline.
15. Add X/social/investor-sentiment adapter with source-quality and manipulation-risk controls.
16. Add market/sector regime features and Fear & Greed-style composite indicator.
17. Run historical event study and walk-forward benchmark; reject weak models.
18. Build 06:00 Telegram report plus intraday candidate/HOLD/EXIT updates.
19. Add paper-trading mode, audit log, monitoring and fail-safe risk limits.
20. Final holdout, reproducibility audit, CI gate and production-readiness check.

Goal: rank the highest expected daily upside opportunities across the eligible BIST universe; tavan is a special high-upside event, not a forced daily outcome.