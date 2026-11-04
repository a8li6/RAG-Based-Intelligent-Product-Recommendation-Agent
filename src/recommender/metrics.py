"""Evaluation metrics as pure functions, with the formulas stated explicitly.

Relevance grades: 2 = strong, 1 = partial, 0 = irrelevant. A product is "relevant" when its
grade is at least 1. All ranking metrics are computed over ELIGIBLE products only.

- Recall@k = |relevant eligible IDs in the top k| / |relevant eligible IDs|
- MRR@k    = 1 / rank of the first relevant result in the top k, else 0
- nDCG@k   = DCG@k / ideal DCG@k with gain 2**grade - 1 and discount log2(rank + 1)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def relevant_eligible(grades: Mapping[str, int], eligible: Iterable[str]) -> set[str]:
    eligible = set(eligible)
    return {pid for pid, grade in grades.items() if grade >= 1 and pid in eligible}


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float | None:
    """None when there is no relevant item (such queries are excluded and counted)."""
    if not relevant:
        return None
    return len(set(ranked[:k]) & relevant) / len(relevant)


def oracle_recall_at_k(relevant_count: int, k: int) -> float | None:
    """Best achievable Recall@k: a ranking that puts relevant items first."""
    if relevant_count == 0:
        return None
    return min(k, relevant_count) / relevant_count


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float | None:
    if not relevant:
        return None
    for rank, pid in enumerate(ranked[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def dcg(grades: Sequence[int]) -> float:
    return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(grades, start=1))


def ndcg_at_k(
    ranked: Sequence[str], grades: Mapping[str, int], eligible: Iterable[str], k: int
) -> float | None:
    """None when no eligible product has a positive grade (zero ideal DCG)."""
    eligible = set(eligible)
    ideal = sorted((g for pid, g in grades.items() if pid in eligible and g > 0), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return None
    gains = [grades.get(pid, 0) if pid in eligible else 0 for pid in ranked[:k]]
    return dcg(gains) / ideal_dcg


def mean_and_count(values: Iterable[float | None]) -> dict:
    """Average over defined values; report how many were used and how many were skipped."""
    values = list(values)
    used = [v for v in values if v is not None]
    return {
        "mean": (sum(used) / len(used)) if used else None,
        "n": len(used),
        "skipped": len(values) - len(used),
    }


def rate(numerator: int, denominator: int) -> dict:
    """A rate with its denominator; N/A (None) rather than a perfect score when it is zero."""
    return {
        "value": (numerator / denominator) if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile, q in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(math.ceil(q / 100 * len(ordered)) - 1, 0)
    return ordered[index]
