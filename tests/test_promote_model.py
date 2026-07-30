from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.tracking.promote_model import (
    _find_latest_pipeline_run,
    _log_sample_recommendations,
    _sample_user_ids,
    main,
    promote_to_production,
    promote_to_staging,
)


@pytest.mark.unit
def test_find_latest_pipeline_run_returns_run_id_of_top_result():
    client = MagicMock()
    run = MagicMock()
    run.info.run_id = "abc123"
    client.search_runs.return_value = [run]

    run_id = _find_latest_pipeline_run(client, "exp-1", "itemknn")

    assert run_id == "abc123"
    client.search_runs.assert_called_once_with(
        experiment_ids=["exp-1"],
        filter_string="tags.source = 'dvc_pipeline' AND tags.model_type = 'itemknn'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )


@pytest.mark.unit
def test_find_latest_pipeline_run_filters_by_given_model_type():
    client = MagicMock()
    run = MagicMock()
    run.info.run_id = "def456"
    client.search_runs.return_value = [run]

    run_id = _find_latest_pipeline_run(client, "exp-1", "svd")

    assert run_id == "def456"
    client.search_runs.assert_called_once_with(
        experiment_ids=["exp-1"],
        filter_string="tags.source = 'dvc_pipeline' AND tags.model_type = 'svd'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )


@pytest.mark.unit
def test_find_latest_pipeline_run_raises_when_no_run_found():
    client = MagicMock()
    client.search_runs.return_value = []

    with pytest.raises(RuntimeError, match="Nenhum run"):
        _find_latest_pipeline_run(client, "exp-1", "itemknn")


@pytest.mark.unit
def test_sample_user_ids_returns_users_known_by_the_model():
    fake_model = MagicMock()
    fake_model._inner_by_user_id = {10: 0, 20: 1, 30: 2, 40: 3}

    with patch("src.tracking.promote_model.joblib.load", return_value=fake_model):
        result = _sample_user_ids("models/model.joblib", n=3)

    assert result == [10, 20, 30]


@pytest.mark.unit
def test_log_sample_recommendations_passes_k_from_settings():
    fake_model = MagicMock()
    fake_model.predict.return_value = [[10, 20], [30, 40]]

    with (
        patch("src.tracking.promote_model.mlflow.pyfunc.load_model", return_value=fake_model),
        patch("src.tracking.promote_model._sample_user_ids", return_value=[1, 2]),
        patch("src.tracking.promote_model.get_settings") as mock_settings,
    ):
        mock_settings.return_value.RECOMMENDATION_K = 10
        _log_sample_recommendations("models:/foo@staging")

    call_args = fake_model.predict.call_args[0][0]
    pd.testing.assert_frame_equal(
        call_args.reset_index(drop=True),
        pd.DataFrame({"user_id": [1, 2], "k": [10, 10]}),
    )


@pytest.mark.unit
def test_promote_to_staging_registers_sets_alias_and_logs_sample():
    fake_model_version = MagicMock(version="1")

    with (
        patch("src.tracking.promote_model.get_settings") as mock_settings,
        patch("src.tracking.promote_model.configure_mlflow_tracking"),
        patch("src.tracking.promote_model.MlflowClient") as mock_client_cls,
        patch("src.tracking.promote_model.mlflow") as mock_mlflow,
        patch(
            "src.tracking.promote_model._find_latest_pipeline_run", return_value="run-1"
        ) as mock_find_run,
        patch("src.tracking.promote_model._log_sample_recommendations") as mock_log_sample,
        patch("src.tracking.promote_model._load_model_config", return_value=("itemknn", {})),
    ):
        mock_settings.return_value.MLFLOW_MODEL_NAME = "ecomm-recsys-itemknn"
        mock_mlflow.register_model.return_value = fake_model_version
        mock_mlflow.tracking.fluent._get_experiment_id.return_value = "exp-1"
        mock_client = mock_client_cls.return_value

        promote_to_staging()

    mock_find_run.assert_called_once_with(mock_client, "exp-1", "itemknn")
    mock_mlflow.register_model.assert_called_once_with(
        "runs:/run-1/model", name="ecomm-recsys-itemknn"
    )
    mock_client.set_registered_model_alias.assert_called_once_with(
        "ecomm-recsys-itemknn", "staging", "1"
    )
    mock_log_sample.assert_called_once_with("models:/ecomm-recsys-itemknn@staging")


@pytest.mark.unit
def test_promote_to_production_promotes_current_staging_version():
    staging_version = MagicMock(version="3")

    with (
        patch("src.tracking.promote_model.get_settings") as mock_settings,
        patch("src.tracking.promote_model.configure_mlflow_tracking"),
        patch("src.tracking.promote_model.MlflowClient") as mock_client_cls,
    ):
        mock_settings.return_value.MLFLOW_MODEL_NAME = "ecomm-recsys-itemknn"
        mock_client = mock_client_cls.return_value
        mock_client.get_model_version_by_alias.return_value = staging_version

        promote_to_production()

    mock_client.get_model_version_by_alias.assert_called_once_with(
        "ecomm-recsys-itemknn", "staging"
    )
    mock_client.set_registered_model_alias.assert_called_once_with(
        "ecomm-recsys-itemknn", "production", "3"
    )


@pytest.mark.unit
def test_main_dispatches_to_staging(monkeypatch):
    monkeypatch.setattr("sys.argv", ["promote_model.py", "--stage", "staging"])
    with patch("src.tracking.promote_model.promote_to_staging") as mock_staging:
        main()
    mock_staging.assert_called_once()


@pytest.mark.unit
def test_main_dispatches_to_production(monkeypatch):
    monkeypatch.setattr("sys.argv", ["promote_model.py", "--stage", "production"])
    with patch("src.tracking.promote_model.promote_to_production") as mock_production:
        main()
    mock_production.assert_called_once()
