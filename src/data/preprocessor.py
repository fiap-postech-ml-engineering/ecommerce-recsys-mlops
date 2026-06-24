"""Pré-processamento de eventos brutos em interações usuário-item ponderadas."""

import pandas as pd
from sklearn.pipeline import FunctionTransformer, Pipeline

EVENT_WEIGHTS = {"view": 1, "addtocart": 2, "transaction": 3}


def build_interactions(events: pd.DataFrame) -> pd.DataFrame:
    """Pondera eventos por tipo e agrega em score por (user_id, item_id).

    Args:
        events: DataFrame com colunas ``user_id``, ``item_id``, ``event``, ``value``,
            ``timestamp`` (saída de ``load_dataset()``), uma linha por evento individual.

    Returns:
        DataFrame com colunas ``user_id``, ``item_id``, ``score``, ``value``, ``timestamp``,
        uma linha por par usuário-item. ``score`` é a soma dos pesos de evento, ``value`` é
        o valor monetário do item e ``timestamp`` é a interação mais recente do par.
    """
    events = events.copy()
    events["score"] = events["event"].map(EVENT_WEIGHTS)
    interactions = events.groupby(["user_id", "item_id"], as_index=False).agg(
        score=("score", "sum"),
        value=("value", "first"),
        timestamp=("timestamp", "max"),
    )
    return interactions


def build_preprocessing_pipeline() -> Pipeline:
    """Monta o pipeline sklearn de pré-processamento de interações.

    Returns:
        Pipeline com um único step que aplica ``build_interactions``, reaproveitável no
        treino e, futuramente, em inferência.
    """
    return Pipeline(steps=[("build_interactions", FunctionTransformer(build_interactions))])
