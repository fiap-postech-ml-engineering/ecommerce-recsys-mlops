"""Isola o carregamento do modelo de produção e a lógica de servir recomendações.

Quando o modelo ainda não está disponível no Registry (ou a decisão de como
scorear candidatos ainda não foi tomada), o serviço opera em modo degradado:
a API sobe normalmente, o healthcheck responde, mas /recommend retorna 503
com uma mensagem clara em vez de derrubar a aplicação inteira.
"""

from __future__ import annotations

import logging

import mlflow

from src.config import get_settings

logger = logging.getLogger(__name__)


class RecommenderService:
    """Carrega o modelo de Production do MLflow Registry e serve recomendações."""

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        """Tenta carregar o modelo em Production. Nunca levanta exceção —
        loga o motivo da falha e deixa o serviço em modo degradado."""
        settings = get_settings()
        model_uri = f"models:/{settings.MLFLOW_MODEL_NAME}@{settings.MLFLOW_MODEL_ALIAS}"

        try:
            self._model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Modelo carregado do Registry: %s", model_uri)
        except Exception:
            self._model = None
            logger.warning(
                "Modelo indisponível em %s — API sobe em modo degradado, "
                "/recommend retornará 503 até o modelo ser promovido.",
                model_uri,
            )

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def recommend(self, user_id: int, k: int) -> list[int]:
        if self._model is None:
            raise RuntimeError("Modelo ainda não disponível em Production.")

        # TODO(quando a estratégia de candidate scoring for definida):
        # o modelo registrado expõe (user_idx, item_idx) -> score (ver
        # docs/internal/mlflow/02_mlflow_boas_praticas.md, seção 5), não
        # recommend(user_id, k) diretamente. Falta decidir contra qual
        # conjunto de itens candidatos rodar o score antes de rankear o top-k.
        raise NotImplementedError(
            "Estratégia de candidate scoring pendente de definição do time."
        )


recommender_service = RecommenderService()