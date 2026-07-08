"""Utilitário compartilhado pelos recommenders baseados na lib `implicit` (ALS/BPR/ItemKNN)."""

import pandas as pd
from scipy.sparse import csr_matrix


def build_user_item_matrix(
    interactions: pd.DataFrame,
) -> tuple[csr_matrix, list[int], dict[int, int], dict[int, int]]:
    """Constrói a matriz esparsa usuário-item e os mapeamentos de índice contíguo.

    A lib `implicit` exige índices 0-based contíguos por linha/coluna — `user_id`/`item_id`
    brutos do RetailRocket são inteiros grandes e esparsos, não usáveis como índice direto.

    Args:
        interactions: DataFrame com colunas `user_id`, `item_id`, `score`.

    Returns:
        Tupla `(user_items, item_ids_by_inner, inner_by_item_id, inner_by_user_id)`.
        `inner_by_user_id` localiza a linha do usuário em `recommend()`.
    """
    user_ids = interactions["user_id"].unique()
    item_ids_by_inner = list(interactions["item_id"].unique())
    inner_by_user_id = {user_id: inner for inner, user_id in enumerate(user_ids)}
    inner_by_item_id = {item_id: inner for inner, item_id in enumerate(item_ids_by_inner)}

    rows = interactions["user_id"].map(inner_by_user_id).to_numpy()
    cols = interactions["item_id"].map(inner_by_item_id).to_numpy()
    data = interactions["score"].to_numpy(dtype="float32")

    user_items = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids_by_inner)))
    return user_items, item_ids_by_inner, inner_by_item_id, inner_by_user_id
