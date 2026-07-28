from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.training.train import _load_model_config, _load_train_data, _persist_model, main


@pytest.mark.unit
def test_load_model_config_reads_type_and_config_from_params_yaml(tmp_path):
    params_path = tmp_path / "params.yaml"
    params_path.write_text("model:\n  type: svd\n  config:\n    n_factors: 10\n")

    model_type, config = _load_model_config(params_path)

    assert model_type == "svd"
    assert config == {"n_factors": 10}


@pytest.mark.unit
def test_load_model_config_defaults_to_itemknn_when_model_key_missing(tmp_path):
    params_path = tmp_path / "params.yaml"
    params_path.write_text("other_stage:\n  some_key: 1\n")

    model_type, config = _load_model_config(params_path)

    assert model_type == "itemknn"
    assert config == {}


@pytest.mark.unit
def test_load_model_config_defaults_config_to_empty_dict_when_missing(tmp_path):
    params_path = tmp_path / "params.yaml"
    params_path.write_text("model:\n  type: popularity\n")

    model_type, config = _load_model_config(params_path)

    assert model_type == "popularity"
    assert config == {}


@pytest.mark.unit
def test_load_train_data_applies_split_filter_and_log_scaling(tmp_path):
    interactions_path = tmp_path / "interactions.parquet"
    raw_interactions = pd.DataFrame({"user_id": [1], "item_id": [2], "score": [3]})
    train_df = pd.DataFrame({"user_id": [1], "item_id": [2], "score": [3]})
    val_df = pd.DataFrame({"user_id": [], "item_id": [], "score": []})
    test_df = pd.DataFrame({"user_id": [], "item_id": [], "score": []})
    scaled_df = pd.DataFrame({"user_id": [1], "item_id": [2], "score": [1.386]})

    with (
        patch("src.training.train.pd.read_parquet", return_value=raw_interactions) as mock_read,
        patch(
            "src.training.train.temporal_split", return_value=(train_df, val_df, test_df)
        ) as mock_split,
        patch(
            "src.training.train.k_core_filter", return_value=(train_df, val_df, test_df)
        ) as mock_filter,
        patch("src.training.train.apply_log_scaling", return_value=scaled_df) as mock_scale,
    ):
        result = _load_train_data("itemknn", interactions_path)

    mock_read.assert_called_once_with(interactions_path)
    mock_split.assert_called_once()
    mock_filter.assert_called_once()
    mock_scale.assert_called_once_with(train_df)
    assert result is scaled_df


@pytest.mark.unit
def test_load_train_data_skips_log_scaling_for_popularity(tmp_path):
    interactions_path = tmp_path / "interactions.parquet"
    raw_interactions = pd.DataFrame({"user_id": [1], "item_id": [2], "score": [3]})
    train_df = pd.DataFrame({"user_id": [1], "item_id": [2], "score": [3]})
    val_df = pd.DataFrame({"user_id": [], "item_id": [], "score": []})
    test_df = pd.DataFrame({"user_id": [], "item_id": [], "score": []})

    with (
        patch("src.training.train.pd.read_parquet", return_value=raw_interactions),
        patch("src.training.train.temporal_split", return_value=(train_df, val_df, test_df)),
        patch("src.training.train.k_core_filter", return_value=(train_df, val_df, test_df)),
        patch("src.training.train.apply_log_scaling") as mock_scale,
    ):
        result = _load_train_data("popularity", interactions_path)

    mock_scale.assert_not_called()
    assert result is train_df


@pytest.mark.unit
def test_persist_model_dumps_via_joblib(tmp_path):
    models_dir = tmp_path / "models"
    model = MagicMock()

    with (
        patch("src.training.train.MODELS_DIR", models_dir),
        patch("src.training.train.joblib.dump") as mock_dump,
    ):
        model_path = _persist_model(model)

    assert model_path == models_dir / "model.joblib"
    assert models_dir.exists()
    mock_dump.assert_called_once_with(model, models_dir / "model.joblib")


@pytest.mark.unit
def test_main_trains_persists_and_logs_to_mlflow_without_notebook_run():
    fake_model = MagicMock()
    fake_model.get_params.return_value = {"model": "itemknn", "n_neighbors": 20}
    train_df = pd.DataFrame({"user_id": [1], "item_id": [2], "score": [1.1]})

    with (
        patch(
            "src.training.train._load_model_config", return_value=("itemknn", {"n_neighbors": 20})
        ) as mock_load_config,
        patch("src.training.train._load_train_data", return_value=train_df) as mock_load_data,
        patch(
            "src.training.train.RecommenderFactory.create", return_value=fake_model
        ) as mock_create,
        patch("src.training.train._persist_model", return_value=Path("models/model.joblib")),
        patch("src.training.train.configure_mlflow_tracking") as mock_configure,
        patch(
            "src.training.train.build_experiment_tags", return_value={"model_type": "itemknn"}
        ) as mock_tags,
        patch("src.training.train.mlflow") as mock_mlflow,
    ):
        main()

    mock_load_config.assert_called_once()
    mock_load_data.assert_called_once()
    mock_create.assert_called_once_with("itemknn", {"n_neighbors": 20})
    fake_model.fit.assert_called_once_with(train_df)

    mock_configure.assert_called_once_with()
    mock_tags.assert_called_once_with(
        model_type="itemknn",
        phase="final",
        dataset_name="retailrocket",
        extra_tags={"source": "dvc_pipeline"},
    )
    mock_mlflow.start_run.assert_called_once_with(tags={"model_type": "itemknn"})
    mock_mlflow.log_params.assert_called_once_with(fake_model.get_params.return_value)
