"""Pré-processamento de eventos brutos em interações usuário-item ponderadas."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.pipeline import FunctionTransformer, Pipeline

from src.data.schema import validate_interactions, validate_raw_events, validate_raw_kaggle_events

EVENT_WEIGHTS = {"view": 1, "addtocart": 2, "transaction": 3}
VALUE_PROPERTY_CODE = "790"


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
                ``value``, ``timestamp`` (saída de ``load_or_build_dataset()``).

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
            ``timestamp`` (saída de ``load_or_build_dataset()``), uma linha por evento
            individual.

    Returns:
        DataFrame com colunas ``user_id``, ``item_id``, ``score``, ``value``, ``timestamp``,
        uma linha por par usuário-item.
    """
    return validate_interactions(WeightedInteractionPreprocessor().transform(events))


def apply_log_scaling(interactions: pd.DataFrame) -> pd.DataFrame:
    """Aplica log1p ao score para atenuar outliers antes do treino de matrix factorization.

    Ver docs/experimentos/0005: score sem cap (máx. 308 vs. mediana 1 no train_df
    pós k-core) alimenta os modelos de matrix factorization/CF implícito (SVD, ALS,
    BPR, ItemKNN) sem limite, instabilizando o treino. Não deve ser usado por
    ``PopularityRecommender`` nem antes da validação do schema Pandera (que exige
    ``score`` inteiro) — só no caminho de treino desses modelos.

    Args:
        interactions: DataFrame com coluna ``score`` (saída de ``build_interactions()``).

    Returns:
        Cópia do DataFrame com ``score`` transformado por ``log1p``.
    """
    df = interactions.copy()
    df["score"] = np.log1p(df["score"])
    return df


def _extract_latest_item_values(item_properties: pd.DataFrame) -> pd.Series:
    """Extrai o valor monetário mais recente por item (property 790).

    Args:
        item_properties: Tabela de propriedades de item (itemid, property, value, timestamp).

    Returns:
        Série indexada por ``itemid`` com o valor monetário mais recente e positivo.
    """
    values = item_properties[item_properties["property"] == VALUE_PROPERTY_CODE].copy()
    values["value"] = pd.to_numeric(
        values["value"].astype(str).str.replace("n", "", regex=False), errors="coerce"
    )
    values = values[values["value"] > 0]
    values = values.sort_values("timestamp").drop_duplicates(subset=["itemid"], keep="last")
    return values.set_index("itemid")["value"]


def build_dataset(events: pd.DataFrame, item_properties: pd.DataFrame) -> pd.DataFrame:
    """Consolida eventos e valores de item no schema bruto do dataset.

    Args:
        events: Tabela de eventos com colunas ``visitorid``, ``itemid``, ``event``, ``datetime``.
        item_properties: Tabela de propriedades de item.

    Returns:
        DataFrame com colunas ``user_id``, ``item_id``, ``event``, ``value``, ``timestamp``.
    """
    validate_raw_kaggle_events(events[["visitorid", "itemid", "event", "datetime"]])
    item_values = _extract_latest_item_values(item_properties)

    dataset = events[["visitorid", "itemid", "event", "datetime"]].copy()
    dataset["value"] = dataset["itemid"].map(item_values)
    dataset = dataset.dropna(subset=["value"])

    dataset = dataset.rename(
        columns={"visitorid": "user_id", "itemid": "item_id", "datetime": "timestamp"}
    )
    dataset = dataset[["user_id", "item_id", "event", "value", "timestamp"]]
    return validate_raw_events(dataset)


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
