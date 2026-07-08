from implicit.als import AlternatingLeastSquares
import numpy as np
import pandas as pd

from src.config import get_settings
from src.models._implicit_utils import build_user_item_matrix
from src.models.base import BaseRecommender


class ALSRecommender(BaseRecommender):
    """Fatoração de matrizes implícita (ALS, via `implicit`) para recomendação.

    Diferente do SVD explícito (`scikit-surprise`), otimiza diretamente feedback
    implícito ponderado por confiança (Hu, Koren & Volinsky, 2008) — ver
    docs/experimentos/0007 para o porquê dessa troca de algoritmo.
    """

    def __init__(self, config: dict | None = None) -> None:
        settings = get_settings()
        self.config = config or {}
        self.factors = self.config.get("factors", settings.ALS_FACTORS)
        self.regularization = self.config.get("regularization", settings.ALS_REGULARIZATION)
        self.alpha = self.config.get("alpha", settings.ALS_ALPHA)
        self.iterations = self.config.get("iterations", settings.ALS_ITERATIONS)
        self.random_state = self.config.get("random_state", settings.RANDOM_SEED)
        self._model: AlternatingLeastSquares | None = None
        self._user_items = None
        self._item_ids_by_inner: list[int] = []
        self._inner_by_user_id: dict[int, int] = {}
        self._seen_items_by_user: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame) -> None:
        """Treina o ALS sobre a matriz esparsa usuário-item de `interactions`."""
        self._user_items, self._item_ids_by_inner, _, self._inner_by_user_id = (
            build_user_item_matrix(interactions)
        )
        self._model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
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
            raise RuntimeError("ALSRecommender.recommend() chamado antes de fit().")

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
            "model": "als",
            "factors": self.factors,
            "regularization": self.regularization,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "random_state": self.random_state,
        }
