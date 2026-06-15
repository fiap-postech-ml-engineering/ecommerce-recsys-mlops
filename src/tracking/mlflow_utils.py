from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess

import mlflow
from mlflow.tracking import MlflowClient

from src.config import LOGS_DIR, MODELS_DIR, get_settings

logger = logging.getLogger(__name__)


def _safe_git(cmd: list[str]) -> str:
    """Execute a git command and return its stdout or a safe fallback.

    Args:
        cmd: Command tokens to execute with subprocess.

    Returns:
        The trimmed stdout value when the command succeeds or "unknown".
    """
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def configure_mlflow_tracking(
    experiment_name: str | None = None,
    backend: str | None = None,
    db_path: str | Path | None = None,
    tracking_uri: str | None = None,
    experiment_tags: dict[str, str] | None = None,
    artifact_root_path: str | Path | None = None,
) -> str:
    """Configure MLflow backend and optionally persist experiment-level tags.

    Args:
        experiment_name: Logical MLflow experiment name. Defaults to
            ``Settings.MLFLOW_EXPERIMENT_NAME``.
        backend: "local" (file-based) ou "databricks". Defaults to "databricks" when
            ``Settings.MLFLOW_TRACKING_URI == "databricks"``, senão "local".
        db_path: Filesystem path to the MLflow file store (backend_store_uri). Only for local.
        tracking_uri: Databricks tracking URI override. Only for databricks.
        experiment_tags: Optional key-value tags to set on the experiment.
        artifact_root_path: Optional filesystem path for experiment artifacts (local only).

    Returns:
        The tracking URI configured for MLflow in the current process.
    """
    settings = get_settings()
    experiment_name = experiment_name or settings.MLFLOW_EXPERIMENT_NAME
    if backend is None:
        backend = "databricks" if settings.MLFLOW_TRACKING_URI == "databricks" else "local"

    if backend == "databricks":
        if settings.DATABRICKS_HOST and "DATABRICKS_HOST" not in os.environ:
            os.environ["DATABRICKS_HOST"] = settings.DATABRICKS_HOST
        if settings.DATABRICKS_TOKEN and "DATABRICKS_TOKEN" not in os.environ:
            os.environ["DATABRICKS_TOKEN"] = settings.DATABRICKS_TOKEN.get_secret_value()

        mlflow.set_tracking_uri(tracking_uri or "databricks")

        try:
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow Databricks conectado")
            logger.info("Experimento: %s", experiment_name)
        except Exception:
            logger.exception("Falha ao configurar experimento MLflow no Databricks")

        return mlflow.get_tracking_uri()

    # Local file-based store
    store_path = Path(db_path) if db_path is not None else LOGS_DIR / "mlruns"
    store_path.mkdir(parents=True, exist_ok=True)
    tracking_uri_local = store_path.resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri_local)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        artifact_root = Path(artifact_root_path) if artifact_root_path else MODELS_DIR / "mlruns"
        artifact_root_dir = artifact_root.resolve()
        artifact_root_dir.mkdir(parents=True, exist_ok=True)
        artifact_location = artifact_root_dir.as_uri()
        client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
            tags=experiment_tags,
        )

    mlflow.set_experiment(experiment_name)

    return tracking_uri_local


def get_git_user() -> str:
    """Return the git user.name configured for the current repository.

    Returns:
        The configured git user name or "unknown" if unavailable.
    """
    return _safe_git(["git", "config", "user.name"])


def build_experiment_tags(
    model_type: str,
    phase: str,
    dataset_name: str,
    dataset_dvc_version: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build standardized tags for grouping and filtering MLflow runs.

    Args:
        model_type: Model identifier (e.g. "popularity", "svd", "mlp").
        phase: Run phase (e.g. "dev", "baseline", "tuning", "final").
        dataset_name: Logical name of the dataset used in this run.
        dataset_dvc_version: DVC revision of the dataset, if tracked.
        extra_tags: Optional extra tags to merge into the resulting payload
            (e.g. "run_group" for development runs).

    Returns:
        Dictionary of tags ready to pass to ``mlflow.start_run(tags=...)``.
    """
    tags = {
        "model_type": model_type,
        "phase": phase,
        "dataset_name": dataset_name,
        "dataset_dvc_version": dataset_dvc_version or "unknown",
        "author": get_git_user(),
    }

    if extra_tags:
        tags.update(extra_tags)

    return tags
