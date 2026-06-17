from __future__ import annotations


def coverage(all_recommended: list[list[int]], catalog_size: int) -> float:
    """Fraction of the catalog covered by the recommendations across all users.

    Args:
        all_recommended: List of recommendation lists, one per user.
        catalog_size: Total number of distinct items in the catalog.

    Returns:
        Coverage in [0, 1].
    """
    if not all_recommended or catalog_size <= 0:
        return 0.0
    unique_items = {item for recs in all_recommended for item in recs}
    return len(unique_items) / catalog_size


def revenue_at_k(
    recommended: list[int],
    relevant_with_value: dict[int, float],
    k: int,
) -> float:
    """Sum of monetary value for recommended items that were actually purchased.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant_with_value: Mapping from item ID to its monetary value for items
            the user purchased (transactions).
        k: Cut-off position.

    Returns:
        Total revenue from hits in the top-K recommendations.
    """
    if not recommended or not relevant_with_value or k <= 0:
        return 0.0
    return sum(
        relevant_with_value[item] for item in recommended[:k] if item in relevant_with_value
    )
