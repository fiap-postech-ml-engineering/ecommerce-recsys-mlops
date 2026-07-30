"""Script de promoção do modelo no MLflow Model Registry (TCF1-134).

Fluxo em duas etapas, cada uma uma invocação explícita via CLI:

- ``--stage staging``: registra a versão do run mais recente do pipeline DVC, seta o
  alias ``staging`` e imprime uma amostra de recomendações para inspeção manual.
- ``--stage production``: promove a versão atualmente em ``staging`` para o alias
  ``production``. Não registra nem valida nada — assume que a validação manual da
  etapa anterior já foi feita por um humano.
"""

import argparse
import logging

import joblib
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from src.config import MODELS_DIR, get_settings
from src.models.base import BaseRecommender
from src.tracking.mlflow_utils import configure_mlflow_tracking
from src.training.train import _load_model_config

logger = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "model.joblib"
SAMPLE_USER_COUNT = 5


def _find_latest_pipeline_run(client: MlflowClient, experiment_id: str, model_type: str) -> str:
    """Localiza o run mais recente gerado pelo pipeline DVC (`train.py`) para `model_type`.

    Args:
        client: Cliente MLflow ativo.
        experiment_id: Experimento onde buscar.
        model_type: Valor de `tags.model_type` a filtrar — o mesmo `model.type` lido de
            `params.yaml` pelo `train.py` (Factory), evitando acoplar este script a um
            modelo específico.

    Returns:
        `run_id` do run mais recente com `tags.source == "dvc_pipeline"` e o
        `model_type` informado.

    Raises:
        RuntimeError: Se nenhum run correspondente for encontrado.
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=(f"tags.source = 'dvc_pipeline' AND tags.model_type = '{model_type}'"),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError(
            f"Nenhum run com tags.source='dvc_pipeline' e tags.model_type='{model_type}' "
            "encontrado. Rode `python -m src.training.train` (ou `dvc repro`) antes de "
            "promover."
        )
    return runs[0].info.run_id


def _sample_user_ids(model_path=MODEL_PATH, n: int = SAMPLE_USER_COUNT) -> list[int]:
    """Amostra `user_id`s conhecidos pelo modelo treinado, para validação manual.

    Usa o `.joblib` local (mesmo artefato que gerou o run promovido) em vez do
    parquet de interações completo — o modelo só conhece o subconjunto de
    usuários que sobreviveu ao split temporal + k-core filtering em
    `_load_train_data` (`src/training/train.py`); usuários fora desse
    subconjunto são cold-start e sempre retornariam lista vazia.

    Args:
        model_path: Caminho do `.joblib` do modelo treinado.
        n: Quantidade de usuários a amostrar.

    Returns:
        Lista dos primeiros `n` `user_id` conhecidos pelo modelo.
    """
    model: BaseRecommender = joblib.load(model_path)
    # Acessa o dict interno diretamente — BaseRecommender (Strategy) não expõe a
    # lista de usuários conhecidos, e adicionar isso à interface obrigaria
    # implementar em todos os modelos só para este script de promoção.
    return [int(user_id) for user_id in list(model._inner_by_user_id.keys())[:n]]


def _log_sample_recommendations(model_uri: str) -> None:
    """Carrega o modelo pyfunc de `model_uri` e loga recomendações de amostra.

    Args:
        model_uri: URI MLflow do modelo (ex: `models:/nome@staging`).
    """
    model = mlflow.pyfunc.load_model(model_uri)
    user_ids = _sample_user_ids()
    k = get_settings().RECOMMENDATION_K
    predictions = model.predict(pd.DataFrame({"user_id": user_ids, "k": [k] * len(user_ids)}))
    for user_id, item_ids in zip(user_ids, predictions, strict=True):
        logger.info("user_id=%s -> recomendações=%s", user_id, item_ids)


def promote_to_staging() -> None:
    """Registra o run vencedor mais recente e promove a versão para o alias `staging`."""
    settings = get_settings()
    configure_mlflow_tracking()
    client = MlflowClient()

    model_type, _ = _load_model_config()
    experiment_id = mlflow.tracking.fluent._get_experiment_id()
    run_id = _find_latest_pipeline_run(client, experiment_id, model_type)

    model_version = mlflow.register_model(f"runs:/{run_id}/model", name=settings.MLFLOW_MODEL_NAME)
    client.set_registered_model_alias(settings.MLFLOW_MODEL_NAME, "staging", model_version.version)
    logger.info(
        "Modelo '%s' versão %s registrado e promovido a staging (run_id=%s).",
        settings.MLFLOW_MODEL_NAME,
        model_version.version,
        run_id,
    )

    _log_sample_recommendations(f"models:/{settings.MLFLOW_MODEL_NAME}@staging")
    logger.info(
        "Validação manual necessária: revise as recomendações acima e, se fizerem "
        "sentido, rode `python -m src.tracking.promote_model --stage production`."
    )


def promote_to_production() -> None:
    """Promove a versão atualmente em `staging` para o alias `production`."""
    settings = get_settings()
    configure_mlflow_tracking()
    client = MlflowClient()

    staging_version = client.get_model_version_by_alias(settings.MLFLOW_MODEL_NAME, "staging")
    client.set_registered_model_alias(
        settings.MLFLOW_MODEL_NAME, "production", staging_version.version
    )
    logger.info(
        "Modelo '%s' versão %s promovido de staging para production.",
        settings.MLFLOW_MODEL_NAME,
        staging_version.version,
    )


def main() -> None:
    """Ponto de entrada CLI: `--stage staging` ou `--stage production`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["staging", "production"], required=True)
    args = parser.parse_args()

    if args.stage == "staging":
        promote_to_staging()
    else:
        promote_to_production()


if __name__ == "__main__":
    main()
