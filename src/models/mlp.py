import pandas as pd

from src.models.base import BaseRecommender


class MLPRecommender(BaseRecommender):
    """Modelo principal baseado em embeddings e MLP (PyTorch)."""

    def fit(self, interactions: pd.DataFrame) -> None:
        raise NotImplementedError

    def recommend(self, user_id: int, k: int) -> list[int]:
        raise NotImplementedError

    def get_params(self) -> dict:
        raise NotImplementedError
