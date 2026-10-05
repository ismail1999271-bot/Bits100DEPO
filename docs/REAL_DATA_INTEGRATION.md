# Bits100 Real Data Integration

The live stack is now split into six real-data domains:

1. BIST market + Level 2/Level 2+ order-book observations
2. KAP disclosures
3. News
4. Fund flow / smart-money exposure
5. Broker consensus
6. Institutional holdings / transactions / research

`LiveDataGateway` normalizes all six into a point-in-time envelope. The
opening-auction engine consumes the market snapshot and calculates indicative
price change, order-book imbalance, queue strength, order arrival/cancellation
flow and volume anomaly.

## Required provider configuration

Set the endpoint variables from the licensed provider contracts:

- `BIST_MARKET_DATA_URL`
- `KAP_API_URL`
- `NEWS_API_URL`
- `FUND_FLOW_API_URL`
- `BROKER_DATA_URL`
- `INSTITUTIONAL_DATA_URL`

An optional `<ENV>_TOKEN` variable is sent as a Bearer token. Credentials must
be stored in GitHub Actions Secrets or the runtime secret manager, never in git.

## BIST Level 2 requirement

The market provider must expose enough fields to map at minimum:

- symbol
- observed_at
- indicative/auction price
- reference/previous-close price
- bid quantity
- ask quantity

For the full opening-pressure model, the feed should additionally expose order
arrival/cancellation data and multi-level depth. The engine never fabricates
missing fields; a missing required auction field is `BLOCKED_MISSING_AUCTION_FIELDS`.

## KAP

KAP's REST API is an authenticated/subscription data publication service. Use
the KAP subscriber endpoint supplied under the organization's contract rather
than scraping the public website.

## Safety

`LIVE_DATA_MODE=fail_closed` and `NO_SYNTHETIC_LIVE_DATA=true` are intentional.
The live pipeline does not place orders. GPT-6 Astra and Claude remain research
and audit components; deterministic quant validation and paper-trading gates
remain mandatory before any production promotion.
