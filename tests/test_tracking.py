import os
from unittest.mock import patch

import pytest

from src.config import get_settings
from src.tracking import configure_mlflow_tracking


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
