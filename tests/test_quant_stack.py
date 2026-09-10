from bist_hunter.auction_trajectory import TrajectoryPoint, build_trajectory
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


def test_opening_trajectory_requires_real_0940_to_0955_schedule():
    points = [
        TrajectoryPoint("2026-09-11T09:40:00+03:00", 60, 2.0),
        TrajectoryPoint("2026-09-11T09:45:00+03:00", 68, 4.0),
        TrajectoryPoint("2026-09-11T09:50:00+03:00", 76, 6.0),
        TrajectoryPoint("2026-09-11T09:55:00+03:00", 84, 8.0),
    ]
    result = build_trajectory("TEST", points)
    assert result.status == "COMPLETE"
    assert result.slope == 8.0
    assert result.acceleration == 0.0


def test_opening_trajectory_does_not_fake_missing_snapshot():
    points = [
        TrajectoryPoint("2026-09-11T09:40:00+03:00", 60, 2.0),
        TrajectoryPoint("2026-09-11T09:50:00+03:00", 76, 6.0),
    ]
    assert build_trajectory("TEST", points).status == "PARTIAL"
