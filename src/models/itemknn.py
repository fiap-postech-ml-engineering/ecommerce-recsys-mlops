from implicit.nearest_neighbours import BM25Recommender
import numpy as np
import pandas as pd

from src.config import get_settings
from src.models._implicit_utils import build_user_item_matrix
from src.models.base import BaseRecommender


class ItemKNNRecommender(BaseRecommender):
    """Vizinhança item-item ponderada por BM25 (via `implicit`) para recomendação.

    Sem fatores latentes — família de algoritmo diferente de SVD/ALS/BPR. Ver
    docs/experimentos/0008: supera as demais mesmo sem tuning neste dataset.
    """

    def __init__(self, config: dict | None = None) -> None:
        settings = get_settings()
        self.config = config or {}
        self.n_neighbors = self.config.get("n_neighbors", settings.ITEMKNN_N_NEIGHBORS)
        self._model: BM25Recommender | None = None
        self._user_items = None
        self._item_ids_by_inner: list[int] = []
        self._inner_by_user_id: dict[int, int] = {}
        self._seen_items_by_user: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame) -> None:
        """Treina o ItemKNN sobre a matriz esparsa usuário-item de `interactions`."""
        self._user_items, self._item_ids_by_inner, _, self._inner_by_user_id = (
            build_user_item_matrix(interactions)
        )
        # BM25Recommender.K = nº de vizinhos; não confundir com Settings.RECOMMENDATION_K
        # (corte de avaliação, usado por todos os modelos).
        self._model = BM25Recommender(K=self.n_neighbors, num_threads=1)
        self._model.fit(self._user_items)
        self._seen_items_by_user = interactions.groupby("user_id")["item_id"].apply(set).to_dict()

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna os top-k item_id recomendados, excluindo itens já vistos.

        Usuários desconhecidos do treino (cold-start) não têm linha na matriz esparsa —
        retorna lista vazia, sem fallback de viés (diferente do SVD explícito). A exclusão
        de itens vistos é reforçada manualmente: `filter_already_liked_items` do
        `BM25Recommender` não é confiável em catálogos pequenos/esparsos — pode deixar
        vazar itens já vistos com score 0 quando não há candidatos suficientes.
        """
        if self._model is None:
            raise RuntimeError("ItemKNNRecommender.recommend() chamado antes de fit().")

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
        return {"model": "itemknn", "n_neighbors": self.n_neighbors}
