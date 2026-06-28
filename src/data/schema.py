"""Schemas pandera para validação do contrato de dados do pipeline."""

import pandas as pd
import pandera.pandas as pa

RAW_EVENTS_SCHEMA = pa.DataFrameSchema(
    {
        "user_id": pa.Column(int, nullable=False),
        "item_id": pa.Column(int, nullable=False),
        "event": pa.Column(
            str,
            pa.Check.isin(["view", "addtocart", "transaction"]),
            nullable=False,
        ),
        "value": pa.Column(float, pa.Check.ge(0), nullable=False),
        "timestamp": pa.Column(pa.DateTime, nullable=False),
    },
    name="RawEventsSchema",
)

INTERACTIONS_SCHEMA = pa.DataFrameSchema(
    {
        "user_id": pa.Column(int, nullable=False),
        "item_id": pa.Column(int, nullable=False),
        "score": pa.Column(int, pa.Check.gt(0), nullable=False),
        "value": pa.Column(float, pa.Check.ge(0), nullable=True),
        "timestamp": pa.Column(pa.DateTime, nullable=False),
    },
    name="InteractionsSchema",
)


def validate_raw_events(df: pd.DataFrame) -> pd.DataFrame:
    """Valida DataFrame bruto de eventos contra RAW_EVENTS_SCHEMA.

    Args:
        df: DataFrame com colunas ``user_id``, ``item_id``, ``event``, ``value``,
            ``timestamp`` (saída de ``load_dataset()``).

    Returns:
        O mesmo DataFrame se válido.

    Raises:
        pandera.errors.SchemaError: Se o schema for violado.
    """
    return RAW_EVENTS_SCHEMA.validate(df)


def validate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Valida DataFrame de interações contra INTERACTIONS_SCHEMA.

    Args:
        df: DataFrame com colunas ``user_id``, ``item_id``, ``score``, ``value``,
            ``timestamp`` (saída de ``build_interactions()``).

    Returns:
        O mesmo DataFrame se válido.

    Raises:
        pandera.errors.SchemaError: Se o schema for violado.
    """
    return INTERACTIONS_SCHEMA.validate(df)
