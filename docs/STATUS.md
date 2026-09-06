# Bits100DEPO implementation status

Updated: 2026-09-06

## Implemented in code

- Package/import structure and CI gates
- Leakage-safe OHLCV normalization and forward event labels
- Tavan/upper-limit event primitives
- Explainable 0-100 candidate scoring
- Daily ranking with an explicit no-quality-signal state
- Cost-aware backtest primitives and chronological walk-forward splits
- Precision/recall, hit-day and drawdown evaluation primitives
- Fund-flow and fund-holding Smart Money exposure proxy
- Unified opportunity quality gate with manipulation-risk rejection
- Provider-neutral HTTP adapters for market data, KAP, news and social feeds
- News/event deduplication and source-quality/manipulation controls
- Market regime / Fear & Greed-style research composite
- Paper-trading ledger, audit events and hard position/order/loss limits
- 06:00-safe morning orchestration and GitHub Actions schedule

## Still requires external market data / credentials

- A compliant/licensed BIST historical/intraday provider must be configured in `BIST_MARKET_DATA_URL`.
- A real KAP feed/adapter must be configured and validated against the selected source.
- Reputable news and X/social providers require their own API/feed access and credentials.
- Telegram delivery requires `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` repository secrets.
- Fund-flow and holdings data must come from a validated source; the Smart Money allocator is an exposure proxy, not transaction truth.

## Production gate

No live-trading claim is made until historical event-study / walk-forward results pass the configured quality thresholds and the final holdout remains untouched until model selection is frozen.
