import pandas as pd

from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Baseline que recomenda os itens mais populares, sem personalização."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self._ranked_items: list[int] = []

    def fit(self, interactions: pd.DataFrame) -> None:
        """Conta interações por item_id e ordena por popularidade decrescente."""
        self._ranked_items = (
            interactions.groupby("item_id")["score"]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna os top-k itens mais populares, ignorando user_id."""
        return self._ranked_items[:k]

    def get_params(self) -> dict:
        return {"model": "popularity"}
