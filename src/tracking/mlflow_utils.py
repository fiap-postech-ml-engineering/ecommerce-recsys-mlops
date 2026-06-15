from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any

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


def build_default_run_tags(
    extra_tags: dict[str, str] | None = None,
) -> dict[str, str]:
    """Create standardized run tags for reproducible experiment tracking.

    Args:
        extra_tags: Optional extra tags to merge into the resulting payload.

    Returns:
        Dictionary of normalized tag values for MLflow runs.
    """
    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    branch = os.getenv("GITHUB_REF_NAME") or _safe_git(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    )
    commit = os.getenv("GITHUB_SHA") or _safe_git(["git", "rev-parse", "HEAD"])

    tags = {
        "runner": "github_actions" if is_ci else "local",
        "ci": str(is_ci).lower(),
        "git_branch": branch,
        "git_commit": commit,
        "python_version": sys.version.split()[0],
        "platform_os": platform.platform(),
        "host": socket.gethostname(),
        "run_timestamp_utc": (
            dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        ),
    }

    if extra_tags:
        tags.update(extra_tags)

    return tags


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize parameters into MLflow-compatible primitive values.

    Args:
        params: Arbitrary parameter dictionary.

    Returns:
        New dictionary with non-primitive values converted to strings.
    """
    normalized: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized
