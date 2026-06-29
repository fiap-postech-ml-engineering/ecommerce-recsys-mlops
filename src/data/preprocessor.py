"""Pré-processamento de eventos brutos em interações usuário-item ponderadas."""

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.pipeline import FunctionTransformer, Pipeline

from src.data.schema import validate_interactions

EVENT_WEIGHTS = {"view": 1, "addtocart": 2, "transaction": 3}


class BasePreprocessor(ABC):
    """Interface Strategy para transformação de eventos brutos em interações user-item."""

    @abstractmethod
    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        """Transforma eventos brutos em interações ponderadas por (user_id, item_id)."""
        ...


class WeightedInteractionPreprocessor(BasePreprocessor):
    """Pondera eventos por tipo e agrega score por (user_id, item_id) via soma.

    Pesos padrão do projeto (view=1, addtocart=2, transaction=3): a soma captura
    padrões de comportamento mais ricos do que usar apenas a interação de maior peso.
    """

    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        """Aplica EVENT_WEIGHTS e agrega interações por par usuário-item.

        Args:
            events: DataFrame com colunas ``user_id``, ``item_id``, ``event``,
                ``value``, ``timestamp`` (saída de ``load_dataset()``).

        Returns:
            DataFrame com colunas ``user_id``, ``item_id``, ``score``, ``value``,
            ``timestamp``, uma linha por par usuário-item. ``score`` é a soma dos
            pesos de evento e ``timestamp`` é a interação mais recente do par.
        """
        df = events.copy()
        df["score"] = df["event"].map(EVENT_WEIGHTS)
        return df.groupby(["user_id", "item_id"], as_index=False).agg(
            score=("score", "sum"),
            value=("value", "first"),
            timestamp=("timestamp", "max"),
        )


def build_interactions(events: pd.DataFrame) -> pd.DataFrame:
    """Pondera eventos por tipo e agrega em score por (user_id, item_id).

    Convenience function que delega para ``WeightedInteractionPreprocessor``.

    Args:
        events: DataFrame com colunas ``user_id``, ``item_id``, ``event``, ``value``,
            ``timestamp`` (saída de ``load_dataset()``), uma linha por evento individual.

    Returns:
        DataFrame com colunas ``user_id``, ``item_id``, ``score``, ``value``, ``timestamp``,
        uma linha por par usuário-item.
    """
    return validate_interactions(WeightedInteractionPreprocessor().transform(events))


def build_preprocessing_pipeline(preprocessor: BasePreprocessor | None = None) -> Pipeline:
    """Monta o pipeline sklearn de pré-processamento de interações.

    Args:
        preprocessor: Estratégia de pré-processamento a usar. Padrão:
            ``WeightedInteractionPreprocessor``.

    Returns:
        Pipeline com um único step que aplica a estratégia escolhida,
        reaproveitável no treino e, futuramente, em inferência.
    """
    strategy = preprocessor or WeightedInteractionPreprocessor()
    return Pipeline(steps=[("preprocess", FunctionTransformer(strategy.transform))])
