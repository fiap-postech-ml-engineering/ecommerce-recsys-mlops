from __future__ import annotations

import math


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of the top-K recommendations that are relevant.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of item IDs the user actually interacted with (transactions).
        k: Cut-off position.

    Returns:
        Precision@K in [0, 1].
    """
    if not recommended or not relevant or k <= 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items that appear in the top-K recommendations.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of item IDs the user actually interacted with (transactions).
        k: Cut-off position.

    Returns:
        Recall@K in [0, 1].
    """
    if not recommended or not relevant or k <= 0:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of item IDs the user actually interacted with (transactions).
        k: Cut-off position.

    Returns:
        NDCG@K in [0, 1].
    """
    if not recommended or not relevant or k <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0.0 else 0.0


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Whether at least one of the top-K recommendations is relevant.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of item IDs the user actually interacted with (transactions).
        k: Cut-off position.

    Returns:
        1.0 if there is at least one hit, 0.0 otherwise.
    """
    if not recommended or not relevant or k <= 0:
        return 0.0
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0
