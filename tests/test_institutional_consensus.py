from datetime import date

from bist_hunter.institutional_consensus import (
    InstitutionalHolding,
    build_position_changes,
    calculate_institutional_consensus,
    consensus_map,
)


def _holding(day, manager, fund, symbol, weight):
    return InstitutionalHolding(day, manager, fund, fund, symbol, weight, 1_000_000_000)


def test_manager_breadth_is_distinct_from_fund_count():
    day = date(2026, 9, 4)
    rows = [
        _holding(day, "AK Portföy", "AK1", "AAA", 0.10),
        _holding(day, "AK Portföy", "AK2", "AAA", 0.05),
        _holding(day, "İş Portföy", "IS1", "AAA", 0.08),
        _holding(day, "Oyak Portföy", "OY1", "AAA", 0.07),
    ]
    result = calculate_institutional_consensus(rows, manager_universe=("AK Portföy", "İş Portföy", "Oyak Portföy"))
    assert result[0].symbol == "AAA"
    assert result[0].manager_count == 3
    assert result[0].fund_count == 4


def test_position_changes_detect_new_and_increases():
    old = [_holding(date(2026, 8, 1), "AK Portföy", "AK1", "AAA", 0.05)]
    new = [
        _holding(date(2026, 9, 1), "AK Portföy", "AK1", "AAA", 0.09),
        _holding(date(2026, 9, 1), "İş Portföy", "IS1", "AAA", 0.04),
    ]
    changes = build_position_changes(old, new)
    statuses = {(c.manager, c.status) for c in changes}
    assert ("AK Portföy", "INCREASE") in statuses
    assert ("İş Portföy", "NEW") in statuses


def test_consensus_score_is_bounded_and_recent_data_wins():
    old_day = date(2026, 6, 1)
    new_day = date(2026, 9, 4)
    rows = [
        _holding(old_day, "AK Portföy", "AK1", "OLD", 0.20),
        _holding(new_day, "AK Portföy", "AK1", "NEW", 0.20),
        _holding(new_day, "İş Portföy", "IS1", "NEW", 0.15),
        _holding(new_day, "Oyak Portföy", "OY1", "NEW", 0.10),
    ]
    result = calculate_institutional_consensus(rows, manager_universe=("AK Portföy", "İş Portföy", "Oyak Portföy"), reference_date=new_day)
    scores = consensus_map(result)
    assert 0 <= scores["NEW"] <= 100
    assert scores["NEW"] > scores["OLD"]
