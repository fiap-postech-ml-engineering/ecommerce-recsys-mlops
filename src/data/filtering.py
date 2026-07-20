"""Filtragem k-core de interações esparsas usuário-item."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _filter_once(
    df: pd.DataFrame, min_user_interactions: int, min_item_interactions: int
) -> pd.DataFrame:
    """Remove uma passada de usuários/itens abaixo do limiar mínimo de interações."""
    user_counts = df["user_id"].value_counts()
    item_counts = df["item_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    valid_items = item_counts[item_counts >= min_item_interactions].index
    return df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]


def _converge_k_core(
    train_df: pd.DataFrame, min_user_interactions: int, min_item_interactions: int
) -> pd.DataFrame:
    """Reaplica `_filter_once` até nenhuma linha a mais ser removida."""
    filtered = train_df
    while True:
        next_filtered = _filter_once(filtered, min_user_interactions, min_item_interactions)
        if len(next_filtered) == len(filtered):
            return next_filtered
        filtered = next_filtered


def k_core_filter(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aplica k-core filtering ao train_df e restringe val/test ao mesmo universo.

    O limiar é calculado apenas sobre train_df para não vazar interações de
    usuários/itens que só aparecem no futuro — consistente com o split temporal
    do projeto. Remover um usuário/item pode derrubar outro abaixo do limiar,
    então o filtro é reaplicado até convergir (nenhuma linha a mais é removida).

    Args:
        train_df: Interações de treino (saída de temporal_split), uma linha por
            par usuário-item.
        val_df: Interações de validação.
        test_df: Interações de teste.
        min_user_interactions: Mínimo de interações (pares distintos com itens)
            por usuário, medido só no treino.
        min_item_interactions: Mínimo de interações (pares distintos com
            usuários) por item, medido só no treino.

    Returns:
        Tupla (train_df, val_df, test_df): train_df contém só o k-core
        resultante; val_df/test_df contêm apenas linhas cujo user_id e item_id
        pertencem a esse k-core.
    """
    filtered_train = _converge_k_core(train_df, min_user_interactions, min_item_interactions)
    valid_users = set(filtered_train["user_id"])
    valid_items = set(filtered_train["item_id"])

    filtered_val = val_df[
        val_df["user_id"].isin(valid_users) & val_df["item_id"].isin(valid_items)
    ]
    filtered_test = test_df[
        test_df["user_id"].isin(valid_users) & test_df["item_id"].isin(valid_items)
    ]

    logger.info(
        "k-core filtering: treino %d -> %d linhas, %d usuários, %d itens",
        len(train_df),
        len(filtered_train),
        len(valid_users),
        len(valid_items),
    )

    return (
        filtered_train.reset_index(drop=True),
        filtered_val.reset_index(drop=True),
        filtered_test.reset_index(drop=True),
    )
