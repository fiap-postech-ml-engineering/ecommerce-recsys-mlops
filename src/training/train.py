"""Script de treino de produção — ponto de entrada do stage "train" do DVC.

Lê `data/processed/interactions.parquet` (saída do stage "preprocess"), treina o
modelo definido em `params.yaml` via `RecommenderFactory` e persiste o artefato em
`models/model.joblib`, registrando os hiperparâmetros no MLflow.
"""

import logging
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml

from src.config import BASE_DIR, MODELS_DIR, get_settings
from src.data.filtering import k_core_filter
from src.data.preprocessor import apply_log_scaling
from src.data.split import temporal_split
from src.models.base import BaseRecommender
from src.models.factory import RecommenderFactory
from src.tracking.mlflow_utils import (
    ItemKNNPyfuncWrapper,
    build_experiment_tags,
    configure_mlflow_tracking,
)

logger = logging.getLogger(__name__)

PARAMS_PATH = BASE_DIR / "params.yaml"
INTERACTIONS_PATH = BASE_DIR / "data" / "processed" / "interactions.parquet"

# apply_log_scaling() não deve ser usada por PopularityRecommender — ver docstring
# em src/data/preprocessor.py.
MODEL_TYPES_WITHOUT_LOG_SCALING = {"popularity"}


def _load_model_config(params_path: Path = PARAMS_PATH) -> tuple[str, dict]:
    """Lê `model.type` e `model.config` de `params.yaml`.

    Args:
        params_path: Caminho para o `params.yaml` do DVC.

    Returns:
        Tupla `(model_type, config)`. `model_type` cai em "itemknn" se a chave
        `model.type` (ou a própria seção `model`) não existir. `config` cai em `{}`
        se `model.config` não existir.
    """
    with params_path.open() as f:
        params = yaml.safe_load(f) or {}

    model_section = params.get("model") or {}
    model_type = model_section.get("type", "itemknn")
    config = model_section.get("config") or {}
    return model_type, config


def _load_train_data(model_type: str, interactions_path: Path = INTERACTIONS_PATH) -> pd.DataFrame:
    """Carrega as interações processadas e prepara o `train_df` para o `fit`.

    Aplica a mesma sequência usada em notebooks/02_experiments.ipynb e
    03_mlp.ipynb: split temporal → k-core filtering (threshold calculado só sobre
    o treino) → log1p scaling do score. Apenas o `train_df` resultante é usado;
    `val_df`/`test_df` não são tocados neste script.

    Args:
        model_type: Tipo de modelo a treinar (de `params.yaml`), usado para decidir
            se o log1p scaling deve ser aplicado.
        interactions_path: Caminho para o parquet de interações pré-processadas.

    Returns:
        `train_df` filtrado e, exceto para `PopularityRecommender`, com score em
        escala log1p, pronto para `model.fit()`.
    """
    settings = get_settings()
    interactions = pd.read_parquet(interactions_path)

    train_df, val_df, test_df = temporal_split(
        interactions,
        test_size=settings.TEST_SIZE,
        validation_size=settings.VALIDATION_SIZE,
    )
    train_df, _, _ = k_core_filter(
        train_df,
        val_df,
        test_df,
        min_user_interactions=settings.MIN_USER_INTERACTIONS,
        min_item_interactions=settings.MIN_ITEM_INTERACTIONS,
    )
    if model_type in MODEL_TYPES_WITHOUT_LOG_SCALING:
        return train_df

    # Score sem cap desestabiliza os modelos de CF implícito (SVD/ALS/BPR/ItemKNN) —
    # ver docs/experimentos/0005. O ItemKNN vencedor (ADR 0010) foi avaliado sobre
    # o score log-escalado, não o bruto.
    return apply_log_scaling(train_df)


def _persist_model(model: BaseRecommender) -> Path:
    """Persiste o modelo treinado via joblib.

    Args:
        model: Instância treinada de `BaseRecommender`.

    Returns:
        Caminho onde o artefato foi salvo.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(model, model_path)
    return model_path


def main() -> None:
    """Treina o modelo de produção definido em `params.yaml` e loga no MLflow."""
    model_type, config = _load_model_config()
    logger.info("Treinando modelo de produção: %s (config=%s)", model_type, config)

    train_df = _load_train_data(model_type)
    model = RecommenderFactory.create(model_type, config)
    model.fit(train_df)

    model_path = _persist_model(model)
    logger.info("Modelo salvo em: %s", model_path)

    configure_mlflow_tracking()
    tags = build_experiment_tags(
        model_type=model_type,
        phase="final",
        dataset_name="retailrocket",
        extra_tags={"source": "dvc_pipeline"},
    )
    settings = get_settings()
    input_example = pd.DataFrame(
        {"user_id": [int(train_df["user_id"].iloc[0])], "k": [settings.RECOMMENDATION_K]}
    )
    with mlflow.start_run(tags=tags):
        mlflow.log_params(model.get_params())
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ItemKNNPyfuncWrapper(),
            artifacts={"model": str(model_path)},
            input_example=input_example,
        )

    logger.info("Treino concluído.")


if __name__ == "__main__":
    main()
