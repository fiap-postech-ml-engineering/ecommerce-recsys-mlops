from implicit.bpr import BayesianPersonalizedRanking
import numpy as np
import pandas as pd

from src.config import get_settings
from src.models._implicit_utils import build_user_item_matrix
from src.models.base import BaseRecommender


class BPRRecommender(BaseRecommender):
    """Bayesian Personalized Ranking (BPR, via `implicit`) para recomendação.

    Otimiza uma loss de ranking pairwise sobre feedback implícito, em vez da loss de
    regressão ponderada por confiança do ALS — ver docs/experimentos/0008 para a
    comparação entre as duas famílias.
    """

    def __init__(self, config: dict | None = None) -> None:
        settings = get_settings()
        self.config = config or {}
        self.factors = self.config.get("factors", settings.BPR_FACTORS)
        self.learning_rate = self.config.get("learning_rate", settings.BPR_LEARNING_RATE)
        self.regularization = self.config.get("regularization", settings.BPR_REGULARIZATION)
        self.iterations = self.config.get("iterations", settings.BPR_ITERATIONS)
        self.random_state = self.config.get("random_state", settings.RANDOM_SEED)
        self._model: BayesianPersonalizedRanking | None = None
        self._user_items = None
        self._item_ids_by_inner: list[int] = []
        self._inner_by_user_id: dict[int, int] = {}
        self._seen_items_by_user: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame) -> None:
        """Treina o BPR sobre a matriz esparsa usuário-item de `interactions`."""
        self._user_items, self._item_ids_by_inner, _, self._inner_by_user_id = (
            build_user_item_matrix(interactions)
        )
        self._model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            num_threads=1,
        )
        self._model.fit(self._user_items)
        self._seen_items_by_user = interactions.groupby("user_id")["item_id"].apply(set).to_dict()

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna os top-k item_id recomendados, excluindo itens já vistos.

        Usuários desconhecidos do treino (cold-start) não têm linha na matriz esparsa —
        retorna lista vazia, sem fallback de viés (diferente do SVD explícito). A exclusão
        de itens vistos é reforçada manualmente (não confia só em
        `filter_already_liked_items`, que em catálogos pequenos/esparsos pode deixar
        vazar itens já vistos com score 0).
        """
        if self._model is None:
            raise RuntimeError("BPRRecommender.recommend() chamado antes de fit().")

        inner_uid = self._inner_by_user_id.get(user_id)
        if inner_uid is None:
            return []

        seen = self._seen_items_by_user.get(user_id, set())
        item_idxs, scores = self._model.recommend(
            inner_uid,
            self._user_items[inner_uid],
            N=k + len(seen),
            filter_already_liked_items=True,
        )
        # Quando não há itens suficientes para preencher N, a lib preenche com índice 0 e
        # score sentinela (float32 mínimo) — precisa ser descartado, não é uma recomendação.
        valid = scores > np.finfo(np.float32).min
        recs = [self._item_ids_by_inner[i] for i in np.asarray(item_idxs)[valid]]
        return [item_id for item_id in recs if item_id not in seen][:k]

    def get_params(self) -> dict:
        return {
            "model": "bpr",
            "factors": self.factors,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "random_state": self.random_state,
        }
