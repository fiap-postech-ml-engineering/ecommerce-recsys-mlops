"""Split temporal de interações em treino, validação e teste."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def temporal_split(
    interactions: pd.DataFrame,
    test_size: float,
    validation_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Particiona interações por ordem cronológica (treino/validação/teste).

    Args:
        interactions: DataFrame de interações com coluna ``timestamp`` (saída de
            ``build_interactions``).
        test_size: Fração mais recente das interações reservada para teste.
        validation_size: Fração intermediária das interações reservada para validação.

    Returns:
        Tupla ``(train_df, val_df, test_df)`` particionada por tempo, sem embaralhar.
    """
    df = interactions.sort_values("timestamp")
    n = len(df)
    train_end = int(n * (1 - test_size - validation_size))
    val_end = int(n * (1 - test_size))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    cold_start_users = set(test_df["user_id"]) - set(train_df["user_id"])
    logger.info(
        "Split temporal: %d treino, %d validação, %d teste, %d usuários cold-start no teste",
        len(train_df),
        len(val_df),
        len(test_df),
        len(cold_start_users),
    )

    return train_df, val_df, test_df
