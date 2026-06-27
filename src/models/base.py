from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    """Interface comum (Strategy) para todos os modelos de recomendação."""

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> None:
        """Treina o modelo a partir do dataframe de interações de treino."""
        ...

    @abstractmethod
    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna os top-k item_id recomendados para o usuário."""
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Retorna os hiperparâmetros do modelo, para logging no MLflow."""
        ...
