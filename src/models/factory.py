from src.models.base import BaseRecommender
from src.models.mlp import MLPRecommender
from src.models.popularity import PopularityRecommender
from src.models.svd import SVDRecommender
from src.tracking.mlflow_utils import ALLOWED_MODEL_TYPES

_REGISTRY: dict[str, type[BaseRecommender]] = {
    "popularity": PopularityRecommender,
    "svd": SVDRecommender,
    "mlp": MLPRecommender,
}


class RecommenderFactory:
    """Factory Method: cria instâncias de BaseRecommender a partir de uma string."""

    @classmethod
    def create(cls, name: str, config: dict) -> BaseRecommender:
        """Instancia o modelo de recomendação correspondente a `name`."""
        if name not in ALLOWED_MODEL_TYPES:
            raise ValueError(f"Modelo desconhecido: {name}")
        return _REGISTRY[name](config)
