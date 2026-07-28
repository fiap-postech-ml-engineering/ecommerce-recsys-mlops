import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluation.evaluate import (
    _load_eval_data,
    _load_model,
    _plot_metrics,
    _save_metrics,
    evaluate_model,
    main,
)


@pytest.mark.unit
def test_load_model_reads_via_joblib(tmp_path):
    model_path = tmp_path / "model.joblib"
    fake_model = MagicMock()

    with patch("src.evaluation.evaluate.joblib.load", return_value=fake_model) as mock_load:
        result = _load_model(model_path)

    mock_load.assert_called_once_with(model_path)
    assert result is fake_model


@pytest.mark.unit
def test_load_eval_data_applies_split_and_k_core_filter(tmp_path):
    interactions_path = tmp_path / "interactions.parquet"
    raw_interactions = pd.DataFrame({"user_id": [1], "item_id": [2], "score": [3]})
    train_df = pd.DataFrame({"user_id": [1], "item_id": [2], "score": [3]})
    val_df = pd.DataFrame({"user_id": [], "item_id": [], "score": []})
    test_df = pd.DataFrame({"user_id": [1], "item_id": [2], "score": [3]})

    with (
        patch(
            "src.evaluation.evaluate.pd.read_parquet", return_value=raw_interactions
        ) as mock_read,
        patch(
            "src.evaluation.evaluate.temporal_split", return_value=(train_df, val_df, test_df)
        ) as mock_split,
        patch(
            "src.evaluation.evaluate.k_core_filter", return_value=(train_df, val_df, test_df)
        ) as mock_filter,
    ):
        result_train, result_test = _load_eval_data(interactions_path)

    mock_read.assert_called_once_with(interactions_path)
    mock_split.assert_called_once()
    mock_filter.assert_called_once()
    assert result_train is train_df
    assert result_test is test_df


@pytest.mark.unit
def test_evaluate_model_computes_the_six_metrics():
    train_df = pd.DataFrame({"user_id": [1, 1, 2, 3], "item_id": [10, 20, 30, 40]})
    test_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 4],
            "item_id": [10, 20, 30, 50],
            "score": [3, 1, 3, 3],
            "value": [5.0, 2.0, 7.0, 1.0],
        }
    )

    model = MagicMock()
    model.recommend.side_effect = lambda user_id, k: {1: [10, 99], 2: [99, 30]}[user_id]

    metrics = evaluate_model(model, test_df, train_df, k=2)

    # eval_users = {1, 2, 3} & {1, 2, 4} = {1, 2} — user 3 has no test interactions,
    # user 4 is cold-start (absent from train_df) and is excluded.
    assert model.recommend.call_count == 2
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["hit_rate"] == pytest.approx(1.0)
    assert metrics["revenue"] == pytest.approx(12.0)
    # catalog_size = itens distintos em train_df pós k-core = {10, 20, 30, 40} = 4
    # recomendações geradas: {10, 99, 30} -> coverage = 3/4
    assert metrics["coverage"] == pytest.approx(0.75)
    assert 0.0 < metrics["ndcg"] < 1.0


@pytest.mark.unit
def test_evaluate_model_skips_users_without_relevant_items():
    train_df = pd.DataFrame({"user_id": [1], "item_id": [10]})
    test_df = pd.DataFrame({"user_id": [1], "item_id": [10], "score": [1], "value": [5.0]})
    model = MagicMock()

    metrics = evaluate_model(model, test_df, train_df, k=2)

    model.recommend.assert_not_called()
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["ndcg"] == 0.0
    assert metrics["hit_rate"] == 0.0
    assert metrics["coverage"] == 0.0
    assert metrics["revenue"] == 0.0


@pytest.mark.unit
def test_save_metrics_writes_flat_json(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics = {
        "precision": 0.5,
        "recall": 0.5,
        "ndcg": 0.5,
        "hit_rate": 1.0,
        "coverage": 0.75,
        "revenue": 12.0,
    }

    _save_metrics(metrics, metrics_path)

    assert json.loads(metrics_path.read_text()) == metrics


@pytest.mark.unit
def test_plot_metrics_saves_a_png(tmp_path):
    metrics = {
        "precision": 0.5,
        "recall": 0.5,
        "ndcg": 0.5,
        "hit_rate": 1.0,
        "coverage": 0.75,
        "revenue": 12.0,
    }

    plot_path = _plot_metrics(metrics, tmp_path)

    assert plot_path == tmp_path / "metrics.png"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


@pytest.mark.unit
def test_main_evaluates_saves_metrics_and_plot():
    fake_model = MagicMock()
    train_df = pd.DataFrame({"user_id": [1], "item_id": [10]})
    test_df = pd.DataFrame({"user_id": [1], "item_id": [10], "score": [3], "value": [5.0]})
    fake_metrics = {
        "precision": 1.0,
        "recall": 1.0,
        "ndcg": 1.0,
        "hit_rate": 1.0,
        "coverage": 1.0,
        "revenue": 5.0,
    }

    with (
        patch("src.evaluation.evaluate._load_model", return_value=fake_model) as mock_load_model,
        patch(
            "src.evaluation.evaluate._load_eval_data", return_value=(train_df, test_df)
        ) as mock_load_data,
        patch(
            "src.evaluation.evaluate.evaluate_model", return_value=fake_metrics
        ) as mock_evaluate,
        patch("src.evaluation.evaluate._save_metrics") as mock_save,
        patch(
            "src.evaluation.evaluate._plot_metrics", return_value=Path("plots/metrics.png")
        ) as mock_plot,
    ):
        main()

    mock_load_model.assert_called_once()
    mock_load_data.assert_called_once()
    mock_evaluate.assert_called_once()
    assert mock_evaluate.call_args.args[0] is fake_model
    assert mock_evaluate.call_args.args[1] is test_df
    assert mock_evaluate.call_args.args[2] is train_df
    mock_save.assert_called_once_with(fake_metrics)
    mock_plot.assert_called_once_with(fake_metrics)
