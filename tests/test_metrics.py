import math

import pytest

from src.metrics import (
    coverage,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    revenue_at_k,
)

# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_precision_at_k_typical():
    assert precision_at_k([1, 2, 3, 4, 5], {2, 4, 6}, k=5) == pytest.approx(2 / 5)


@pytest.mark.unit
def test_precision_at_k_zero_hits():
    assert precision_at_k([1, 2, 3], {7, 8}, k=3) == 0.0


@pytest.mark.unit
def test_precision_at_k_empty_recommended():
    assert precision_at_k([], {1, 2}, k=5) == 0.0


@pytest.mark.unit
def test_precision_at_k_empty_relevant():
    assert precision_at_k([1, 2, 3], set(), k=3) == 0.0


@pytest.mark.unit
def test_precision_at_k_k_larger_than_list():
    # Only 3 items recommended, k=10 — truncation should not error
    assert precision_at_k([1, 2, 3], {1, 2}, k=10) == pytest.approx(2 / 10)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_recall_at_k_typical():
    assert recall_at_k([1, 2, 3, 4, 5], {2, 4, 6}, k=5) == pytest.approx(2 / 3)


@pytest.mark.unit
def test_recall_at_k_zero_hits():
    assert recall_at_k([1, 2, 3], {7, 8}, k=3) == 0.0


@pytest.mark.unit
def test_recall_at_k_empty_recommended():
    assert recall_at_k([], {1, 2}, k=5) == 0.0


@pytest.mark.unit
def test_recall_at_k_empty_relevant():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


@pytest.mark.unit
def test_recall_at_k_k_larger_than_list():
    assert recall_at_k([1, 2], {1, 2, 3}, k=10) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ndcg_at_k_perfect_ranking():
    # All hits at position 0 and 1 → DCG == IDCG
    result = ndcg_at_k([1, 2, 3], {1, 2}, k=3)
    assert result == pytest.approx(1.0)


@pytest.mark.unit
def test_ndcg_at_k_hit_at_end_lower_than_hit_at_start():
    score_early = ndcg_at_k([1, 99], {1}, k=2)
    score_late = ndcg_at_k([99, 1], {1}, k=2)
    assert score_early > score_late


@pytest.mark.unit
def test_ndcg_at_k_zero_hits():
    assert ndcg_at_k([1, 2, 3], {7, 8}, k=3) == 0.0


@pytest.mark.unit
def test_ndcg_at_k_empty_recommended():
    assert ndcg_at_k([], {1, 2}, k=5) == 0.0


@pytest.mark.unit
def test_ndcg_at_k_empty_relevant():
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


@pytest.mark.unit
def test_ndcg_at_k_typical():
    # hit at position 0: 1/log2(2)=1; hit at position 2: 1/log2(4)=0.5
    # IDCG (2 ideal hits): 1/log2(2)+1/log2(3)
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k([1, 99, 2], {1, 2}, k=3) == pytest.approx(dcg / idcg)


# ---------------------------------------------------------------------------
# hit_rate_at_k
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_hit_rate_at_k_hit():
    assert hit_rate_at_k([1, 2, 3], {3, 5}, k=3) == 1.0


@pytest.mark.unit
def test_hit_rate_at_k_no_hit():
    assert hit_rate_at_k([1, 2, 3], {7, 8}, k=3) == 0.0


@pytest.mark.unit
def test_hit_rate_at_k_hit_outside_k():
    # item 3 is in relevant but beyond k=2
    assert hit_rate_at_k([1, 2, 3], {3}, k=2) == 0.0


@pytest.mark.unit
def test_hit_rate_at_k_empty_recommended():
    assert hit_rate_at_k([], {1}, k=5) == 0.0


@pytest.mark.unit
def test_hit_rate_at_k_empty_relevant():
    assert hit_rate_at_k([1, 2], set(), k=2) == 0.0


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_coverage_typical():
    recs = [[1, 2], [2, 3], [4, 5]]
    assert coverage(recs, catalog_size=10) == pytest.approx(5 / 10)


@pytest.mark.unit
def test_coverage_full():
    recs = [[1, 2, 3]]
    assert coverage(recs, catalog_size=3) == pytest.approx(1.0)


@pytest.mark.unit
def test_coverage_empty_recs():
    assert coverage([], catalog_size=10) == 0.0


@pytest.mark.unit
def test_coverage_zero_catalog():
    assert coverage([[1, 2]], catalog_size=0) == 0.0


# ---------------------------------------------------------------------------
# revenue_at_k
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_revenue_at_k_typical():
    result = revenue_at_k([1, 2, 3, 4], {2: 10.0, 4: 5.0}, k=4)
    assert result == pytest.approx(15.0)


@pytest.mark.unit
def test_revenue_at_k_hit_outside_k():
    # item 3 is relevant but beyond k=2
    result = revenue_at_k([1, 2, 3], {3: 50.0}, k=2)
    assert result == pytest.approx(0.0)


@pytest.mark.unit
def test_revenue_at_k_no_hits():
    assert revenue_at_k([1, 2, 3], {7: 100.0}, k=3) == pytest.approx(0.0)


@pytest.mark.unit
def test_revenue_at_k_empty_recommended():
    assert revenue_at_k([], {1: 10.0}, k=5) == pytest.approx(0.0)


@pytest.mark.unit
def test_revenue_at_k_empty_relevant():
    assert revenue_at_k([1, 2], {}, k=2) == pytest.approx(0.0)
