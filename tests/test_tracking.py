import datetime as dt
import os
from unittest.mock import patch

import mlflow
from mlflow.tracking import MlflowClient
import pytest

from src.config import get_settings
from src.tracking import (
    ALLOWED_MODEL_TYPES,
    build_experiment_tags,
    configure_mlflow_tracking,
    log_evaluation_metrics,
    start_notebook_run,
)
from src.tracking.mlflow_utils import (
    _build_run_name,
    _get_or_create_parent_run,
    _validate_notebook_run_inputs,
)


def test_configure_mlflow_tracking_uses_settings_for_databricks(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "databricks")
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "fake-token")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "/Users/test/exp")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    get_settings.cache_clear()

    with (
        patch("src.tracking.mlflow_utils.mlflow.set_tracking_uri") as mock_set_uri,
        patch("src.tracking.mlflow_utils.mlflow.set_experiment") as mock_set_experiment,
        patch(
            "src.tracking.mlflow_utils.mlflow.get_tracking_uri",
            return_value="databricks",
        ),
    ):
        result = configure_mlflow_tracking()

    mock_set_uri.assert_called_once_with("databricks")
    mock_set_experiment.assert_called_once_with("/Users/test/exp")
    assert result == "databricks"
    assert os.environ["DATABRICKS_HOST"] == "https://example.cloud.databricks.com"
    assert os.environ["DATABRICKS_TOKEN"] == "fake-token"

    get_settings.cache_clear()


@pytest.mark.integration
@pytest.mark.slow
def test_configure_mlflow_tracking_authenticates_with_databricks():
    settings = get_settings()
    if not settings.DATABRICKS_HOST or not settings.DATABRICKS_TOKEN:
        pytest.skip("DATABRICKS_HOST/DATABRICKS_TOKEN não configurados em .env")

    tracking_uri = configure_mlflow_tracking()
    assert tracking_uri == "databricks"

    client = MlflowClient()
    experiments = client.search_experiments(max_results=1)
    assert experiments is not None


def test_build_experiment_tags_includes_expected_keys_and_defaults(monkeypatch):
    monkeypatch.setattr("src.tracking.mlflow_utils._safe_git", lambda cmd: "test-user")

    tags = build_experiment_tags(
        model_type="mlp",
        phase="dev",
        dataset_name="retailrocket",
    )

    assert tags["model_type"] == "mlp"
    assert tags["phase"] == "dev"
    assert tags["dataset_name"] == "retailrocket"
    assert tags["dataset_dvc_version"] == "unknown"
    assert tags["author"] == "test-user"


def test_build_experiment_tags_merges_extra_and_dvc_version(monkeypatch):
    monkeypatch.setattr("src.tracking.mlflow_utils._safe_git", lambda cmd: "test-user")

    tags = build_experiment_tags(
        model_type="mlp",
        phase="dev",
        dataset_name="retailrocket",
        dataset_dvc_version="a1b2c3d",
        extra_tags={"run_group": "notebook_mlp_dev"},
    )

    assert tags["dataset_dvc_version"] == "a1b2c3d"
    assert tags["run_group"] == "notebook_mlp_dev"


def test_build_experiment_tags_run_group_first_class(monkeypatch):
    monkeypatch.setattr("src.tracking.mlflow_utils._safe_git", lambda cmd: "test-user")

    tags = build_experiment_tags(
        model_type="svd",
        phase="baseline",
        dataset_name="retailrocket",
        run_group="svd_sem_fe_nf50",
    )

    assert tags["run_group"] == "svd_sem_fe_nf50"


def test_build_experiment_tags_extra_tags_override_run_group(monkeypatch):
    monkeypatch.setattr("src.tracking.mlflow_utils._safe_git", lambda cmd: "test-user")

    tags = build_experiment_tags(
        model_type="svd",
        phase="baseline",
        dataset_name="retailrocket",
        run_group="a",
        extra_tags={"run_group": "b"},
    )

    assert tags["run_group"] == "b"


def test_build_experiment_tags_no_run_group_key_when_not_set(monkeypatch):
    monkeypatch.setattr("src.tracking.mlflow_utils._safe_git", lambda cmd: "test-user")

    tags = build_experiment_tags(model_type="svd", phase="dev", dataset_name="retailrocket")

    assert "run_group" not in tags


# --- Validação de entradas ---


def test_validate_model_type_raises_on_invalid():
    with pytest.raises(ValueError, match="inválido"):
        _validate_notebook_run_inputs("xgboost", "sem_fe")


def test_validate_model_type_accepts_all_allowed():
    for model_type in ALLOWED_MODEL_TYPES:
        _validate_notebook_run_inputs(model_type, "sem_fe")


def test_validate_test_name_raises_on_uppercase():
    with pytest.raises(ValueError, match="inválidos"):
        _validate_notebook_run_inputs("svd", "Sem_Fe")


def test_validate_test_name_raises_on_space():
    with pytest.raises(ValueError, match="inválidos"):
        _validate_notebook_run_inputs("svd", "sem fe")


def test_validate_test_name_raises_on_special_chars():
    with pytest.raises(ValueError, match="inválidos"):
        _validate_notebook_run_inputs("svd", "sem/fe")


def test_validate_test_name_accepts_valid_patterns():
    for name in ("sem-fe", "nf50", "sem_fe_nf50", "com-fe-nf100"):
        _validate_notebook_run_inputs("svd", name)


# --- Geração de run_name ---


def test_build_run_name_format(monkeypatch):
    fixed_dt = dt.datetime(2026, 6, 16, 14, 32, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(
        "src.tracking.mlflow_utils.dt.datetime",
        type("FakeDatetime", (), {"now": staticmethod(lambda tz=None: fixed_dt)})(),
    )

    name = _build_run_name("svd", "sem_fe_nf50")

    assert name == "svd_sem_fe_nf50_20260616T143200Z"


# --- Hierarquia parent/child (MLflow local) ---


@pytest.fixture()
def local_mlflow(tmp_path):
    """Fixture que configura MLflow local isolado para testes de hierarquia."""
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-exp")
    client = MlflowClient(tracking_uri)
    exp = client.get_experiment_by_name("test-exp")
    yield {"client": client, "experiment_id": exp.experiment_id}
    mlflow.set_tracking_uri("")


def test_get_or_create_parent_run_creates(local_mlflow):
    run_id = _get_or_create_parent_run("svd", local_mlflow["experiment_id"])

    assert isinstance(run_id, str)
    assert len(run_id) > 0


def test_get_or_create_parent_run_reuses(local_mlflow):
    run_id_1 = _get_or_create_parent_run("svd", local_mlflow["experiment_id"])
    run_id_2 = _get_or_create_parent_run("svd", local_mlflow["experiment_id"])

    assert run_id_1 == run_id_2


def test_start_notebook_run_tags_and_name(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-exp")

    monkeypatch.setattr("src.tracking.mlflow_utils._safe_git", lambda cmd: "test-user")
    monkeypatch.setattr(
        "src.tracking.mlflow_utils.get_settings",
        lambda: type("S", (), {"MLFLOW_EXPERIMENT_NAME": "test-exp"})(),
    )

    with start_notebook_run("svd", "sem_fe_nf50", phase="baseline", dataset_name="retailrocket"):
        active = mlflow.active_run()
        assert active.data.tags["run_group"] == "svd_sem_fe_nf50"
        assert active.data.tags["model_type"] == "svd"
        assert active.data.tags["phase"] == "baseline"
        assert active.info.run_name.startswith("svd_sem_fe_nf50_")

    mlflow.set_tracking_uri("")


def test_start_notebook_run_parent_hierarchy(tmp_path, monkeypatch):
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("test-exp")

    monkeypatch.setattr("src.tracking.mlflow_utils._safe_git", lambda cmd: "test-user")
    monkeypatch.setattr(
        "src.tracking.mlflow_utils.get_settings",
        lambda: type("S", (), {"MLFLOW_EXPERIMENT_NAME": "test-exp"})(),
    )

    client = MlflowClient()
    exp = client.get_experiment_by_name("test-exp")

    with start_notebook_run(
        "svd", "sem_fe_nf50", phase="baseline", dataset_name="retailrocket"
    ) as run:
        child_run_id = run.info.run_id

    all_runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(all_runs) == 3  # parent1, parent2, child

    child = client.get_run(child_run_id)
    parent2_id = child.data.tags["mlflow.parentRunId"]
    parent2 = client.get_run(parent2_id)
    parent1_id = parent2.data.tags["mlflow.parentRunId"]
    parent1 = client.get_run(parent1_id)

    assert parent2.info.run_name == "svd_sem_fe_nf50"
    assert parent1.info.run_name == "svd"
    assert "mlflow.parentRunId" not in parent1.data.tags

    mlflow.set_tracking_uri("")


@pytest.mark.integration
@pytest.mark.slow
def test_full_dummy_run_logs_to_mlflow_with_expected_tags():
    settings = get_settings()
    if not settings.DATABRICKS_HOST or not settings.DATABRICKS_TOKEN:
        pytest.skip("DATABRICKS_HOST/DATABRICKS_TOKEN não configurados em .env")

    class DummyRecommender:
        """Minimal stand-in for a BaseRecommender, used to validate tracking."""

        def get_params(self) -> dict:
            return {"k": 10, "hidden_dims": [128, 64]}

    configure_mlflow_tracking()
    tags = build_experiment_tags(
        model_type="dummy",
        phase="dev",
        dataset_name="retailrocket_sample",
        extra_tags={"run_group": "tracking_validation"},
    )

    dummy = DummyRecommender()
    run_name = f"dummy_dev_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.set_tag(
            "mlflow.note.content",
            "Run de validação da infraestrutura de tracking MLflow/Databricks.",
        )
        mlflow.log_params(dummy.get_params())
        mlflow.log_metrics(
            {
                "model.precision_at_k": 0.0,
                "model.recall_at_k": 0.0,
                "model.ndcg_at_k": 0.0,
                "model.hit_rate_at_k": 0.0,
            }
        )

    client = MlflowClient()
    logged_run = client.get_run(run.info.run_id)
    assert logged_run.data.tags["model_type"] == "dummy"
    assert logged_run.data.tags["run_group"] == "tracking_validation"
    assert logged_run.data.params["k"] == "10"


# --- log_evaluation_metrics ---


def test_log_evaluation_metrics_maps_short_keys(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-metrics")

    with mlflow.start_run():
        log_evaluation_metrics(
            {
                "precision": 0.14,
                "recall": 0.09,
                "ndcg": 0.17,
                "hit_rate": 0.28,
                "coverage": 0.38,
                "revenue": 980.50,
            },
            k=10,
        )
        run = mlflow.active_run()
        # Forçar flush consultando o run diretamente
        client = MlflowClient()
        data = client.get_run(run.info.run_id).data

    assert data.metrics["model.precision_at_k"] == pytest.approx(0.14)
    assert data.metrics["model.recall_at_k"] == pytest.approx(0.09)
    assert data.metrics["model.ndcg_at_k"] == pytest.approx(0.17)
    assert data.metrics["model.hit_rate_at_k"] == pytest.approx(0.28)
    assert data.metrics["business.coverage"] == pytest.approx(0.38)
    assert data.metrics["business.revenue_at_k"] == pytest.approx(980.50)
    assert data.params["recommendation_k"] == "10"

    mlflow.set_tracking_uri("")


def test_log_evaluation_metrics_passes_already_prefixed_keys(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-metrics")

    with mlflow.start_run():
        log_evaluation_metrics({"model.precision_at_k": 0.20, "business.coverage": 0.50}, k=5)
        run = mlflow.active_run()
        client = MlflowClient()
        data = client.get_run(run.info.run_id).data

    assert data.metrics["model.precision_at_k"] == pytest.approx(0.20)
    assert data.metrics["business.coverage"] == pytest.approx(0.50)

    mlflow.set_tracking_uri("")


def test_log_evaluation_metrics_unknown_keys_pass_through(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-metrics")

    with mlflow.start_run():
        log_evaluation_metrics({"model.train_loss": 0.034}, k=10)
        run = mlflow.active_run()
        client = MlflowClient()
        data = client.get_run(run.info.run_id).data

    assert data.metrics["model.train_loss"] == pytest.approx(0.034)

    mlflow.set_tracking_uri("")


def test_start_notebook_run_logs_params_when_provided(tmp_path, monkeypatch):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-exp")

    monkeypatch.setattr("src.tracking.mlflow_utils._safe_git", lambda cmd: "test-user")
    monkeypatch.setattr(
        "src.tracking.mlflow_utils.get_settings",
        lambda: type("S", (), {"MLFLOW_EXPERIMENT_NAME": "test-exp"})(),
    )

    with start_notebook_run(
        "svd",
        "com_params",
        phase="dev",
        dataset_name="retailrocket",
        params={"n_factors": 50, "random_state": 42},
    ) as run:
        run_id = run.info.run_id

    data = MlflowClient().get_run(run_id).data
    assert data.params["n_factors"] == "50"
    assert data.params["random_state"] == "42"

    mlflow.set_tracking_uri("")


def test_start_notebook_run_no_params_does_not_log_params(tmp_path, monkeypatch):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    mlflow.set_experiment("test-exp")

    monkeypatch.setattr("src.tracking.mlflow_utils._safe_git", lambda cmd: "test-user")
    monkeypatch.setattr(
        "src.tracking.mlflow_utils.get_settings",
        lambda: type("S", (), {"MLFLOW_EXPERIMENT_NAME": "test-exp"})(),
    )

    with start_notebook_run("svd", "sem_params", phase="dev", dataset_name="retailrocket") as run:
        run_id = run.info.run_id

    data = MlflowClient().get_run(run_id).data
    assert data.params == {}

    mlflow.set_tracking_uri("")
