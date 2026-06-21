"""Smoke test manual do tracking MLflow contra o workspace Databricks.

Para cada um dos 3 experimentos lógicos documentados em `02_mlflow_boas_praticas.md`
(notebook_baselines_training, notebook_mlp_training, Production), cria 2 modelos
mockados (popularity, svd) x 2 testes cada (smoke, edge_case), usando
`start_notebook_run` para validar a hierarquia de runs, tags e métricas. Não depende
de `src/models/` ou `src/evaluation/` (ainda stubs vazios) — usa dados e modelos
mockados.

Pré-requisito: `.env` com MLFLOW_TRACKING_URI=databricks, DATABRICKS_HOST e
DATABRICKS_TOKEN configurados (ver `01_configuracao_mlflow.md`).

Uso:
    python docs/internal/mlflow/03_smoke_test_runs.py
"""

from __future__ import annotations

import os
from typing import Any

from src.config import get_settings
from src.metrics import (
    coverage,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    revenue_at_k,
)
from src.tracking import configure_mlflow_tracking, log_evaluation_metrics, start_notebook_run

RECOMMENDATION_K = 10
CATALOG_SIZE = 100

EXPERIMENT_SUFFIXES = [
    "02 - ECOMM_RECSYS - notebook_baselines_training",
    "02 - ECOMM_RECSYS - notebook_mlp_training",
    "02 - ECOMM_RECSYS - Production",
]


class DummyPopularityModel:
    """Modelo mockado representando a estratégia de popularidade."""

    def get_params(self) -> dict[str, Any]:
        return {"strategy": "most_popular", "k": RECOMMENDATION_K}


class DummySVDModel:
    """Modelo mockado representando a estratégia SVD."""

    def get_params(self) -> dict[str, Any]:
        return {"n_factors": 50, "k": RECOMMENDATION_K}


def _mock_dataset(test_name: str) -> dict[str, Any]:
    """Build mocked recommendation data for a given test variation.

    Args:
        test_name: "smoke" (caso feliz) ou "edge_case" (sem recomendações/relevantes).

    Returns:
        Dict com `recommended`, `relevant`, `relevant_with_value` e `all_recommended`.
    """
    if test_name == "edge_case":
        return {
            "recommended": [],
            "relevant": set(),
            "relevant_with_value": {},
            "all_recommended": [[]],
        }

    recommended = list(range(1, RECOMMENDATION_K + 1))
    relevant = {2, 4, 6}
    return {
        "recommended": recommended,
        "relevant": relevant,
        "relevant_with_value": {item: 19.9 * item for item in relevant},
        "all_recommended": [recommended],
    }


def _compute_metrics(dataset: dict[str, Any]) -> dict[str, float]:
    """Compute the 6 model/business metrics for a mocked dataset."""
    recommended = dataset["recommended"]
    relevant = dataset["relevant"]
    return {
        "precision": precision_at_k(recommended, relevant, RECOMMENDATION_K),
        "recall": recall_at_k(recommended, relevant, RECOMMENDATION_K),
        "ndcg": ndcg_at_k(recommended, relevant, RECOMMENDATION_K),
        "hit_rate": hit_rate_at_k(recommended, relevant, RECOMMENDATION_K),
        "coverage": coverage(dataset["all_recommended"], CATALOG_SIZE),
        "revenue": revenue_at_k(recommended, dataset["relevant_with_value"], RECOMMENDATION_K),
    }


MODELS: dict[str, Any] = {
    "popularity": DummyPopularityModel(),
    "svd": DummySVDModel(),
}
TEST_NAMES = ["smoke", "edge_case"]


def _experiment_paths() -> list[str]:
    """Build the 3 absolute Databricks workspace paths from the configured base."""
    base_dir = get_settings().MLFLOW_EXPERIMENT_NAME
    return [f"{base_dir}/{suffix}" for suffix in EXPERIMENT_SUFFIXES]


def main() -> None:
    runs_created = 0
    for experiment_name in _experiment_paths():
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
        get_settings.cache_clear()
        tracking_uri = configure_mlflow_tracking()

        for model_type, model in MODELS.items():
            for test_name in TEST_NAMES:
                dataset = _mock_dataset(test_name)
                metrics = _compute_metrics(dataset)
                with start_notebook_run(
                    model_type=model_type,
                    test_name=test_name,
                    phase="dev",
                    dataset_name="mock",
                    params=model.get_params(),
                ):
                    log_evaluation_metrics(metrics, k=RECOMMENDATION_K)
                runs_created += 1
                print(f"[ok] {experiment_name} :: {model_type}/{test_name} -> métricas: {metrics}")

        print(f"Tracking URI: {tracking_uri} | Experimento: {experiment_name}\n")

    print(f"Runs criadas no total: {runs_created}")


if __name__ == "__main__":
    main()
