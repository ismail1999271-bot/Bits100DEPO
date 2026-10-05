from bist_hunter.opening_auction import AuctionSnapshot, score_opening_auction


def test_strong_buy_auction_gets_signal():
    result = score_opening_auction(
        AuctionSnapshot(
            symbol="TEST",
            observed_at="2026-09-10T09:45:00+03:00",
            indicative_price=110,
            reference_price=100,
            bid_qty=900_000,
            ask_qty=100_000,
            order_arrival_buy=500_000,
            order_arrival_sell=100_000,
            order_cancel_buy=20_000,
            order_cancel_sell=10_000,
            volume=200_000,
            baseline_volume=50_000,
            tavan_dna=85,
            kap_score=80,
            news_score=75,
            fund_score=70,
            institutional_score=80,
            broker_score=80,
        )
    )
    assert result.status == "SIGNAL"
    assert result.score is not None and result.score > 70
    assert result.imbalance_pct is not None and result.imbalance_pct > 70
    assert "strong_buy_imbalance" in result.reasons


def test_missing_market_data_is_blocked():
    result = score_opening_auction(
        AuctionSnapshot(
            symbol="TEST",
            observed_at="2026-09-10T09:45:00+03:00",
            indicative_price=0,
            reference_price=100,
            bid_qty=100,
            ask_qty=100,
        )
    )
    assert result.status == "BLOCKED_INVALID_PRICE"
    assert result.score is None
