# Bits100DEPO implementation status

Updated: 2026-09-07

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
- Institutional data fabric for market, fundamentals, estimates, holdings, flows, research, transactions and alternative data
- Institutional Consensus across broker/manager views
- Broker recommendation consensus scoring
- Broker technical follow-up fields: entry range, target, stop and risk/reward
- Performance leaderboard with rank change and momentum acceleration
- News/event deduplication and source-quality/manipulation controls
- Market regime / Fear & Greed-style research composite
- Paper-trading ledger, audit events and hard position/order/loss limits
- 06:00-safe morning orchestration and GitHub Actions schedule
- Claude Technology Scout runtime
- Scout -> Quant Researcher -> Signal Auditor automated research-loop workflow
- Leakage-safe experiment feedback ledger for agent research memory
- Fail-closed production readiness gates and CLI

## External activation still required

The repository now has the wiring/contracts for all requested production layers, but real credentials/data cannot be fabricated or supplied by the codebase:

- A compliant/licensed BIST historical/intraday provider must be configured in `BIST_MARKET_DATA_URL` / related secrets.
- A real KAP feed/adapter must be configured and validated against the selected source.
- Fund-flow and holdings data must come from a validated source; Smart Money remains an exposure proxy, not transaction truth.
- Broker Consensus requires a real normalized broker-research feed in `BROKER_DATA_URL`.
- Institutional Intelligence requires a real authorized provider in `INSTITUTIONAL_DATA_URL`.
- Claude research automation requires `ANTHROPIC_API_KEY`.
- Telegram delivery requires `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

## Validation / training

- Historical training workflow requests up to ~10 years of BIST bars and already exceeds the requested 5–6 year minimum when the provider returns sufficient history.
- Walk-forward/backtest primitives are present, but final performance numbers must be produced from real historical data and a frozen final holdout.
- Paper trading is real-market-data based and persists state/metrics, but remains strictly non-live-order execution.
- Agent feedback records completed experiment results only; it cannot alter production rules directly.

## Production gate

Live trading is **BLOCKED by design** until all required providers are configured, historical event-study/walk-forward results pass configured quality thresholds, model selection is frozen, and the final holdout remains untouched. The system must fail closed rather than invent missing data.
