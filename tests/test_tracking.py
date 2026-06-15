import os
from unittest.mock import patch

import pytest

from src.config import get_settings
from src.tracking import build_experiment_tags, configure_mlflow_tracking


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

    from mlflow.tracking import MlflowClient

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


@pytest.mark.integration
@pytest.mark.slow
def test_full_dummy_run_logs_to_mlflow_with_expected_tags():
    settings = get_settings()
    if not settings.DATABRICKS_HOST or not settings.DATABRICKS_TOKEN:
        pytest.skip("DATABRICKS_HOST/DATABRICKS_TOKEN não configurados em .env")

    import datetime as dt

    import mlflow
    from mlflow.tracking import MlflowClient

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
