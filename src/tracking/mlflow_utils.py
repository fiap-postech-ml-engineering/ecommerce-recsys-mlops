from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import datetime as dt
import logging
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from src.config import LOGS_DIR, MODELS_DIR, get_settings

logger = logging.getLogger(__name__)

ALLOWED_MODEL_TYPES: frozenset[str] = frozenset({"popularity", "svd", "mlp"})


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

        if "DATABRICKS_HOST" not in os.environ or "DATABRICKS_TOKEN" not in os.environ:
            logger.warning(
                "MLFLOW_TRACKING_URI=databricks mas DATABRICKS_HOST/DATABRICKS_TOKEN não "
                "estão configurados. Veja docs/internal/configuracao_mlflow.md."
            )

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
    run_group: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build standardized tags for grouping and filtering MLflow runs.

    Args:
        model_type: Model identifier (e.g. "popularity", "svd", "mlp").
        phase: Run phase (e.g. "dev", "baseline", "tuning", "final").
        dataset_name: Logical name of the dataset used in this run.
        dataset_dvc_version: DVC revision of the dataset, if tracked.
        run_group: Logical group name for aggregating related runs (e.g. "svd_sem_fe_nf50").
            Prefer using ``start_notebook_run``, which derives this automatically.
        extra_tags: Optional extra tags to merge into the resulting payload. Keys in
            ``extra_tags`` take precedence over explicitly named parameters.

    Returns:
        Dictionary of tags ready to pass to ``mlflow.start_run(tags=...)``.
    """
    tags: dict[str, str] = {
        "model_type": model_type,
        "phase": phase,
        "dataset_name": dataset_name,
        "dataset_dvc_version": dataset_dvc_version or "unknown",
        "author": get_git_user(),
    }

    if run_group is not None:
        tags["run_group"] = run_group

    if extra_tags:
        tags.update(extra_tags)

    return tags


def _validate_notebook_run_inputs(model_type: str, test_name: str) -> None:
    """Validate model_type and test_name before creating a notebook run.

    Args:
        model_type: Must be one of ALLOWED_MODEL_TYPES.
        test_name: Must match ``[a-z0-9_-]+`` (lowercase, digits, hyphens, underscores).

    Raises:
        ValueError: If either argument fails validation.
    """
    if model_type not in ALLOWED_MODEL_TYPES:
        allowed = sorted(ALLOWED_MODEL_TYPES)
        raise ValueError(f"model_type '{model_type}' inválido. Valores permitidos: {allowed}")
    if not re.fullmatch(r"[a-z0-9_-]+", test_name):
        raise ValueError(
            f"test_name '{test_name}' contém caracteres inválidos. "
            "Use apenas letras minúsculas, números, hífens e underscores (ex: 'sem-fe-nf50')."
        )


def _build_run_name(model_type: str, test_name: str) -> str:
    """Build a unique run name with a UTC timestamp suffix.

    Args:
        model_type: Model identifier.
        test_name: Test variation descriptor.

    Returns:
        Run name in the format ``<model_type>_<test_name>_<YYYYmmddTHHMMSSZ>``.
    """
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_type}_{test_name}_{timestamp}"


def _get_or_create_parent_run(
    run_name: str,
    experiment_id: str,
    tags: dict[str, str] | None = None,
) -> str:
    """Return the run_id of an existing organizational parent run, or create one.

    Uses search_runs to avoid duplicates. Not safe against concurrent creation
    by multiple processes using the same run_name at the same instant.

    Args:
        run_name: Exact name to look up or create.
        experiment_id: MLflow experiment to search within.
        tags: Tags to apply when creating a new run (ignored if run already exists).

    Returns:
        The run_id (str) of the found or created run.
    """
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        max_results=1,
    )
    if runs:
        return runs[0].info.run_id
    run = client.create_run(experiment_id, run_name=run_name, tags=tags)
    return run.info.run_id


_METRIC_KEY_MAP: dict[str, str] = {
    "precision": "model.precision_at_k",
    "recall": "model.recall_at_k",
    "ndcg": "model.ndcg_at_k",
    "hit_rate": "model.hit_rate_at_k",
    "coverage": "business.coverage",
    "revenue": "business.revenue_at_k",
}


def log_evaluation_metrics(metrics: dict[str, float], k: int) -> None:
    """Log evaluation metrics to the active MLflow run with standardized names.

    Accepts short keys (``"precision"``, ``"recall"``, ``"ndcg"``,
    ``"hit_rate"``, ``"coverage"``, ``"revenue"``) or already-prefixed keys
    (``"model.precision_at_k"``). Unknown keys are passed through unchanged.
    Requires an active MLflow run.

    Args:
        metrics: Mapping of metric name to value.
        k: Cut-off used during evaluation — logged as ``recommendation_k`` parameter.
    """
    mlflow.log_param("recommendation_k", k)
    normalized = {_METRIC_KEY_MAP.get(key, key): value for key, value in metrics.items()}
    mlflow.log_metrics(normalized)


@contextmanager
def start_notebook_run(
    model_type: str,
    test_name: str,
    phase: str,
    dataset_name: str,
    dataset_dvc_version: str | None = None,
    extra_tags: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Open a child MLflow run inside a two-level organizational hierarchy.

    Creates or reuses two parent runs (``model_type`` and
    ``model_type_test_name``) and yields the child run with auto-derived
    ``run_name`` and ``run_group`` tag. Call ``configure_mlflow_tracking``
    before using this function.

    Args:
        model_type: Model type. Must be one of ``ALLOWED_MODEL_TYPES``.
        test_name: Short descriptor for this test variation. Only ``[a-z0-9_-]``.
        phase: Run phase: ``"dev"``, ``"baseline"``, ``"tuning"`` or ``"final"``.
        dataset_name: Logical dataset name (e.g. ``"retailrocket"``).
        dataset_dvc_version: DVC revision of the dataset, if tracked.
        extra_tags: Additional tags merged into the run (take precedence over defaults).
        params: Model hyperparameters logged via ``mlflow.log_params`` on run open.

    Yields:
        The active MLflow child run.

    Raises:
        ValueError: If ``model_type`` or ``test_name`` are invalid.
        RuntimeError: If the active experiment is not found (forgot to call
            ``configure_mlflow_tracking``).
    """
    _validate_notebook_run_inputs(model_type, test_name)

    run_group = f"{model_type}_{test_name}"
    run_name = _build_run_name(model_type, test_name)
    tags = build_experiment_tags(
        model_type=model_type,
        phase=phase,
        dataset_name=dataset_name,
        dataset_dvc_version=dataset_dvc_version,
        run_group=run_group,
        extra_tags=extra_tags,
    )

    exp_name = get_settings().MLFLOW_EXPERIMENT_NAME
    experiment = MlflowClient().get_experiment_by_name(exp_name)
    if experiment is None:
        raise RuntimeError(
            f"Experimento '{exp_name}' não encontrado. "
            "Chame configure_mlflow_tracking() antes de start_notebook_run()."
        )

    experiment_id = experiment.experiment_id
    parent1_id = _get_or_create_parent_run(model_type, experiment_id)
    parent2_id = _get_or_create_parent_run(
        run_group, experiment_id, {"mlflow.parentRunId": parent1_id}
    )

    with mlflow.start_run(run_name=run_name, tags=tags, parent_run_id=parent2_id) as run:
        if params is not None:
            mlflow.log_params(params)
        yield run
