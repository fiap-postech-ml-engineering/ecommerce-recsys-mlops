from unittest.mock import patch

import pandas as pd
import pytest

from src.data.preprocess import _save_interactions, main


@pytest.mark.unit
def test_save_interactions_writes_parquet(tmp_path):
    interactions_path = tmp_path / "processed" / "interactions.parquet"
    interactions = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [2],
            "score": [3],
            "value": [10.0],
            "timestamp": pd.to_datetime(["2026-01-01"]),
        }
    )

    result_path = _save_interactions(interactions, interactions_path)

    assert result_path == interactions_path
    assert interactions_path.exists()
    assert pd.read_parquet(interactions_path).equals(interactions)


@pytest.mark.unit
def test_main_loads_builds_and_saves_interactions():
    events = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [2],
            "event": ["view"],
            "value": [10.0],
            "timestamp": pd.to_datetime(["2026-01-01"]),
        }
    )
    interactions = pd.DataFrame({"user_id": [1], "item_id": [2], "score": [1]})

    with (
        patch("src.data.preprocess.load_or_build_dataset", return_value=events) as mock_load,
        patch("src.data.preprocess.build_interactions", return_value=interactions) as mock_build,
        patch("src.data.preprocess._save_interactions") as mock_save,
    ):
        main()

    mock_load.assert_called_once_with()
    mock_build.assert_called_once_with(events)
    mock_save.assert_called_once_with(interactions)
