from bist_hunter.provider_readiness import evaluate_provider_readiness, production_data_ready


def test_provider_readiness_is_fail_closed_without_credentials():
    result = evaluate_provider_readiness({})
    assert len(result) == 6
    assert all(item.status == "BLOCKED_MISSING_ENDPOINT" for item in result)
    assert production_data_ready({}) is False


def test_provider_readiness_requires_credentials():
    env = {"BIST_MARKET_DATA_URL": "https://provider.example/market"}
    result = evaluate_provider_readiness(env)
    market = next(item for item in result if item.domain == "market")
    assert market.status == "BLOCKED_MISSING_CREDENTIAL"
    assert production_data_ready(env) is False


def test_provider_readiness_accepts_configured_domain():
    env = {
        "BIST_MARKET_DATA_URL": "https://provider.example/market",
        "BIST_MARKET_DATA_TOKEN": "secret",
    }
    market = next(item for item in evaluate_provider_readiness(env) if item.domain == "market")
    assert market.configured is True
    assert market.credentialed is True
    assert market.status == "CONFIGURED"
