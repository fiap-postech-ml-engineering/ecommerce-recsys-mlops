"""Dataset PyTorch de interações usuário-item para treino do MLPRecommender."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset


class InteractionsDataset(Dataset):
    """Dataset de triplas (user_idx, item_idx, rating) para o MLPRecommender.

    Recebe um DataFrame já processado/splitado (saída de ``split.py``), com
    colunas ``user_id``, ``item_id``, ``score``. Como o dataset original usa
    IDs esparsos, mapeia ``user_id``/``item_id`` para índices contíguos
    (0-indexed) — necessário para a camada de embedding em ``mlp.py``.
    """

    def __init__(self, interactions: pd.DataFrame) -> None:
        user_ids = sorted(interactions["user_id"].unique())
        item_ids = sorted(interactions["item_id"].unique())

        self.user_id_to_idx: dict[int, int] = {uid: idx for idx, uid in enumerate(user_ids)}
        self.item_id_to_idx: dict[int, int] = {iid: idx for idx, iid in enumerate(item_ids)}

        self.n_users = len(user_ids)
        self.n_items = len(item_ids)

        self._user_idx = interactions["user_id"].map(self.user_id_to_idx).to_numpy()
        self._item_idx = interactions["item_id"].map(self.item_id_to_idx).to_numpy()
        self._ratings = interactions["score"].to_numpy(dtype="float32")

    def __len__(self) -> int:
        return len(self._ratings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self._user_idx[idx], dtype=torch.long),
            torch.tensor(self._item_idx[idx], dtype=torch.long),
            torch.tensor(self._ratings[idx], dtype=torch.float32),
        )
