import pandas as pd

from src.models.base import BaseRecommender


class SVDRecommender(BaseRecommender):
    """Baseline de fatoração de matrizes (SVD) para recomendação."""

    def fit(self, interactions: pd.DataFrame) -> None:
        raise NotImplementedError

    def recommend(self, user_id: int, k: int) -> list[int]:
        raise NotImplementedError

    def get_params(self) -> dict:
        raise NotImplementedError
