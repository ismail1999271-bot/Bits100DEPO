# Bits100DEPO implementation status

Updated: 2026-09-16

## Implemented in code

- Package/import structure and CI gates
- Leakage-safe OHLCV normalization and forward event labels
- Tavan/upper-limit event primitives
- Leakage-safe Tavan-DNA feature engineering and chronological model training
- Three-way historical Tavan-DNA backtest runner: train / validation / untouched holdout
- Expanded Tavan-DNA walk-forward validation with an untouched final holdout
- E2E Tavan-DNA probability -> Quant Score -> auction trajectory ranking path
- Cost-aware backtest primitives and chronological walk-forward splits
- Precision/recall, expectancy, hit-rate, drawdown and risk-adjusted validation primitives
- Explainable 0-100 candidate scoring with coverage-aware confidence
- Daily ranking and opening-auction trajectory at 09:40 / 09:45 / 09:50 / 09:55
- Single BIST100+ universe construction and global scan interface
- Fund-flow and fund-holding Smart Money exposure proxy
- Unified opportunity quality gate with manipulation-risk rejection
- Provider-neutral HTTP adapters for market data, KAP, news and social feeds
- Strict point-in-time market/event provider contracts and duplicate/staleness rejection
- Provider readiness matrix that fails closed when endpoints or credentials are missing
- Institutional data fabric for market, fundamentals, estimates, holdings, flows, research, transactions and alternative data
- Institutional Consensus across broker/manager views
- Broker recommendation consensus scoring
- Broker technical follow-up fields: entry range, target, stop and risk/reward
- Performance leaderboard with rank change and momentum acceleration
- News/event deduplication and source-quality/manipulation controls
- Market regime / Fear & Greed-style research composite
- Paper-trading ledger with deterministic spread/slippage execution, audit events and hard limits
- 06:00-safe morning orchestration and GitHub Actions schedule
- Claude Technology Scout and GPT-6 Astra research runtimes
- Scout -> Quant Researcher -> Signal Auditor automated research-loop workflow
- Leakage-safe experiment feedback ledger for agent research memory
- Fail-closed production readiness gates and CLI

## External activation still required

The code is wired for real providers, but credentials and licensed access are external prerequisites and are never fabricated by the repository:

- BIST Level-2/Level-2+ historical/intraday feed: configure a compliant licensed provider and its endpoint/credential.
- KAP: configure and validate a real subscribed KAP-compatible feed.
- News: configure a real provider and normalize its point-in-time publication timestamps.
- Fund flow / holdings: configure a validated source; Smart Money remains a proxy until source truth is available.
- Broker / institutional: configure real normalized research and holdings/transaction feeds.
- GPT-6 Astra / Claude: configure the corresponding API secrets for research automation.
- Telegram: configure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` only after the research/paper pipeline is validated.

## Validation / training

The repository contains the real historical execution path: provider CSV -> validation -> Tavan-DNA features -> chronological walk-forward -> untouched final holdout -> cost-aware metrics. No performance claim is valid until this runner is executed on real historical BIST data.

The live auction path is fail-closed: licensed Level-2/2+ observations are validated before entering the 09:40/09:45/09:50/09:55 trajectory. Missing indicative price/order-book data cannot become a synthetic signal.

## Production gate

Automatic exchange order placement is intentionally absent. The system remains a research, signal-ranking and paper-trading platform. Production readiness requires real provider health, data integrity, walk-forward validation, an untouched final holdout and sufficient paper-trading evidence. Missing prerequisites keep the gate BLOCKED.
