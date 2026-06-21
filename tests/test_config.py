from src.config import Settings


def test_settings_does_not_require_databricks_credentials(monkeypatch):
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.delenv("DATABRICKS_TOKEN", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    settings = Settings(_env_file=None)

    assert settings.MLFLOW_TRACKING_URI == "databricks"
    assert settings.DATABRICKS_HOST is None
    assert settings.DATABRICKS_TOKEN is None
