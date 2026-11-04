import math

import pytest

from recommender.metrics import (
    dcg,
    mean_and_count,
    mrr_at_k,
    ndcg_at_k,
    oracle_recall_at_k,
    percentile,
    rate,
    recall_at_k,
    relevant_eligible,
)

GRADES = {"A": 2, "B": 1, "C": 2, "D": 0, "X": 2}  # X is relevant but not eligible
ELIGIBLE = ["A", "B", "C", "D", "E"]


def test_relevant_eligible_excludes_ineligible_and_zero_grades():
    assert relevant_eligible(GRADES, ELIGIBLE) == {"A", "B", "C"}


# --- Recall@k ------------------------------------------------------------------------


def test_recall_counts_relevant_found_in_the_top_k():
    relevant = {"A", "B", "C"}
    assert recall_at_k(["A", "D", "B", "E"], relevant, 10) == pytest.approx(2 / 3)
    assert recall_at_k(["A", "D", "B", "E"], relevant, 1) == pytest.approx(1 / 3)
    assert recall_at_k(["D", "E"], relevant, 10) == 0.0
    assert recall_at_k(["A", "B", "C"], relevant, 3) == 1.0


def test_recall_ignores_items_after_k_and_duplicates_do_not_inflate():
    assert recall_at_k(["D", "E", "A"], {"A"}, 2) == 0.0
    assert recall_at_k(["A", "A", "A"], {"A", "B"}, 10) == 0.5


def test_recall_is_undefined_without_relevant_items():
    assert recall_at_k(["A"], set(), 10) is None


def test_oracle_recall_caps_at_one_and_shows_the_ceiling():
    assert oracle_recall_at_k(4, 10) == 1.0
    assert oracle_recall_at_k(20, 10) == 0.5
    assert oracle_recall_at_k(0, 10) is None


# --- MRR@k ---------------------------------------------------------------------------


def test_mrr_is_the_reciprocal_rank_of_the_first_relevant_result():
    relevant = {"B"}
    assert mrr_at_k(["B", "A"], relevant, 3) == 1.0
    assert mrr_at_k(["A", "B"], relevant, 3) == 0.5
    assert mrr_at_k(["A", "D", "B"], relevant, 3) == pytest.approx(1 / 3)
    assert mrr_at_k(["A", "D", "E", "B"], relevant, 3) == 0.0  # beyond k
    assert mrr_at_k(["A"], set(), 3) is None


# --- nDCG@k --------------------------------------------------------------------------


def test_dcg_uses_gain_2_pow_grade_minus_1_and_log2_rank_plus_1():
    assert dcg([2]) == pytest.approx(3.0)  # (2^2 - 1) / log2(2)
    assert dcg([0, 1]) == pytest.approx(1 / math.log2(3))
    assert dcg([2, 1, 2]) == pytest.approx(3 / 1 + 1 / math.log2(3) + 3 / 2)


def test_ndcg_perfect_ranking_is_one():
    # Eligible graded items: A=2, C=2, B=1. Ideal top 3 is A, C, B.
    assert ndcg_at_k(["A", "C", "B"], GRADES, ELIGIBLE, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["C", "A", "B"], GRADES, ELIGIBLE, 3) == pytest.approx(1.0)


def test_ndcg_penalizes_a_worse_order_with_hand_computed_value():
    ideal = dcg([2, 2, 1])
    got = dcg([1, 0, 2])  # B, D, A
    assert ndcg_at_k(["B", "D", "A"], GRADES, ELIGIBLE, 3) == pytest.approx(got / ideal)
    assert 0 < got / ideal < 1


def test_ndcg_ideal_uses_only_eligible_items():
    # X has grade 2 but is not eligible: it must not enter the ideal or the gain.
    value = ndcg_at_k(["X", "A", "C"], GRADES, ELIGIBLE, 3)
    expected = dcg([0, 2, 2]) / dcg([2, 2, 1])
    assert value == pytest.approx(expected)


def test_ndcg_is_undefined_when_the_ideal_is_zero():
    assert ndcg_at_k(["A"], {"A": 0, "B": 0}, ["A", "B"], 3) is None
    assert ndcg_at_k(["A"], {"X": 2}, ["A"], 3) is None


def test_ndcg_with_fewer_than_k_results():
    assert ndcg_at_k(["A"], GRADES, ELIGIBLE, 3) == pytest.approx(dcg([2]) / dcg([2, 2, 1]))
    assert ndcg_at_k([], GRADES, ELIGIBLE, 3) == 0.0


# --- aggregation ---------------------------------------------------------------------


def test_mean_and_count_reports_denominators_and_skips():
    assert mean_and_count([1.0, None, 0.0, 0.5, None]) == {"mean": 0.5, "n": 3, "skipped": 2}
    assert mean_and_count([None]) == {"mean": None, "n": 0, "skipped": 1}
    assert mean_and_count([]) == {"mean": None, "n": 0, "skipped": 0}


def test_rate_is_not_applicable_when_the_denominator_is_zero():
    assert rate(0, 0) == {"value": None, "numerator": 0, "denominator": 0}  # N/A, not a perfect 0
    assert rate(3, 4) == {"value": 0.75, "numerator": 3, "denominator": 4}


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        ([1, 2, 3, 4], 50, 2),
        ([1, 2, 3, 4], 95, 4),
        ([5], 50, 5),
        ([3, 1, 2], 100, 3),
        ([3, 1, 2], 1, 1),
    ],
)
def test_nearest_rank_percentile(values, q, expected):
    assert percentile(values, q) == expected


def test_percentile_of_nothing_is_none():
    assert percentile([], 50) is None
