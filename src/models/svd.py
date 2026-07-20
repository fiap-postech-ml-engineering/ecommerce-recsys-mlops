import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader

from src.config import get_settings
from src.models.base import BaseRecommender


class SVDRecommender(BaseRecommender):
    """Baseline de fatoração de matrizes (SVD, via scikit-surprise) para recomendação."""

    def __init__(self, config: dict | None = None) -> None:
        settings = get_settings()
        self.config = config or {}
        self.n_factors = self.config.get("n_factors", settings.SVD_N_FACTORS)
        self.n_epochs = self.config.get("n_epochs", settings.SVD_N_EPOCHS)
        self.lr_all = self.config.get("lr_all", 0.005)
        self.reg_all = self.config.get("reg_all", 0.02)
        self.random_state = self.config.get("random_state", settings.RANDOM_SEED)
        self._algo: SVD | None = None
        self._trainset = None
        self._item_ids_by_inner: list[int] = []
        self._inner_by_item_id: dict[int, int] = {}
        self._seen_items_by_user: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame) -> None:
        """Treina o SVD sobre (user_id, item_id, score) e memoriza itens já vistos por usuário.

        Guarda o `trainset` e os fatores latentes treinados para pontuar recomendações via
        numpy em `recommend`, em vez de chamar `algo.predict` item a item (inviável no
        catálogo de ~180 mil itens do RetailRocket).
        """
        reader = Reader(rating_scale=(interactions["score"].min(), interactions["score"].max()))
        dataset = Dataset.load_from_df(interactions[["user_id", "item_id", "score"]], reader)
        self._trainset = dataset.build_full_trainset()

        self._algo = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=self.random_state,
        )
        self._algo.fit(self._trainset)

        self._item_ids_by_inner = [
            self._trainset.to_raw_iid(inner) for inner in range(self._trainset.n_items)
        ]
        self._inner_by_item_id = {
            item_id: inner for inner, item_id in enumerate(self._item_ids_by_inner)
        }
        self._seen_items_by_user = interactions.groupby("user_id")["item_id"].apply(set).to_dict()

    def _score_all_items(self, user_id: int) -> np.ndarray:
        """Pontua todo o catálogo para `user_id`, replicando a fórmula de estimativa do SVD.

        Usuários desconhecidos do treino (cold-start) recaem no viés global + viés do item
        (sem termo personalizado), mesmo comportamento padrão do ``scikit-surprise``.
        """
        try:
            inner_uid = self._trainset.to_inner_uid(user_id)
        except ValueError:
            return self._trainset.global_mean + self._algo.bi
        user_bias = self._trainset.global_mean + self._algo.bu[inner_uid]
        return user_bias + self._algo.bi + self._algo.qi @ self._algo.pu[inner_uid]

    def recommend(self, user_id: int, k: int) -> list[int]:
        """Retorna os top-k item_id com maior score estimado, excluindo itens já vistos."""
        if self._algo is None or self._trainset is None:
            raise RuntimeError("SVDRecommender.recommend() chamado antes de fit().")

        scores = self._score_all_items(user_id).copy()
        for item_id in self._seen_items_by_user.get(user_id, set()):
            inner = self._inner_by_item_id.get(item_id)
            if inner is not None:
                scores[inner] = -np.inf

        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top_k_unordered = np.argpartition(-scores, k - 1)[:k]
        top_k = top_k_unordered[np.argsort(-scores[top_k_unordered])]
        return [self._item_ids_by_inner[inner] for inner in top_k]

    def get_params(self) -> dict:
        return {
            "model": "svd",
            "n_factors": self.n_factors,
            "n_epochs": self.n_epochs,
            "lr_all": self.lr_all,
            "reg_all": self.reg_all,
            "random_state": self.random_state,
        }
