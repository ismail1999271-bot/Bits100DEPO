from bist_hunter.quant_score import calculate_quant_score
from bist_hunter.walk_forward import final_holdout, walk_forward_splits


def test_quant_score_renormalizes_available_components():
    result = calculate_quant_score({"auction": 90, "tavan_dna": 80})
    assert result.status == "SIGNAL"
    assert result.score == 84.44
    assert result.confidence == 0.45


def test_quant_score_blocks_low_coverage():
    result = calculate_quant_score({"auction": 90}, min_coverage=0.50)
    assert result.status == "BLOCKED_INSUFFICIENT_DATA"


def test_walk_forward_is_strictly_chronological():
    splits = walk_forward_splits(20, 10, 5)
    assert splits[0].train == tuple(range(10))
    assert splits[0].test == tuple(range(10, 15))
    assert max(splits[0].train) < min(splits[0].test)


def test_final_holdout_is_untouched_tail():
    split = final_holdout(20, 5)
    assert split.train == tuple(range(15))
    assert split.test == tuple(range(15, 20))
