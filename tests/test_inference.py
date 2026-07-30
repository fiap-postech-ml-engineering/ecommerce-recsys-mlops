from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.api.inference import RecommenderService


@pytest.mark.unit
def test_is_ready_false_before_load():
    service = RecommenderService()

    assert service.is_ready is False


@pytest.mark.unit
def test_load_sets_model_when_registry_succeeds():
    fake_model = MagicMock()
    service = RecommenderService()

    with (
        patch("src.api.inference.configure_mlflow_tracking"),
        patch("src.api.inference.mlflow.pyfunc.load_model", return_value=fake_model),
    ):
        service.load()

    assert service.is_ready is True


@pytest.mark.unit
def test_load_keeps_degraded_mode_when_registry_fails():
    service = RecommenderService()

    with (
        patch("src.api.inference.configure_mlflow_tracking"),
        patch("src.api.inference.mlflow.pyfunc.load_model", side_effect=RuntimeError("boom")),
    ):
        service.load()

    assert service.is_ready is False


@pytest.mark.unit
def test_recommend_raises_when_model_not_loaded():
    service = RecommenderService()

    with pytest.raises(RuntimeError, match="não disponível em Production"):
        service.recommend(user_id=1, k=5)


@pytest.mark.unit
def test_recommend_builds_dataframe_and_returns_first_row():
    fake_model = MagicMock()
    fake_model.predict.return_value = [[10, 20, 30]]
    service = RecommenderService()
    service._model = fake_model

    result = service.recommend(user_id=42, k=3)

    assert result == [10, 20, 30]
    call_args = fake_model.predict.call_args[0][0]
    pd.testing.assert_frame_equal(call_args, pd.DataFrame({"user_id": [42], "k": [3]}))
