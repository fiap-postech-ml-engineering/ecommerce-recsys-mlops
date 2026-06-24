import pandas as pd

from src.data.split import temporal_split


def _make_interactions(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": list(range(n)),
            "item_id": list(range(n)),
            "score": [1] * n,
            "value": [10.0] * n,
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )


def test_temporal_split_respects_60_20_20_proportions():
    interactions = _make_interactions(100)

    train_df, val_df, test_df = temporal_split(interactions, test_size=0.2, validation_size=0.2)

    assert len(train_df) == 60
    assert len(val_df) == 20
    assert len(test_df) == 20


def test_temporal_split_preserves_chronological_order_without_overlap():
    interactions = _make_interactions(100)

    train_df, val_df, test_df = temporal_split(interactions, test_size=0.2, validation_size=0.2)

    assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    assert val_df["timestamp"].max() <= test_df["timestamp"].min()


def test_temporal_split_is_deterministic():
    interactions = _make_interactions(100).sample(frac=1, random_state=1)

    train1, val1, test1 = temporal_split(interactions, test_size=0.2, validation_size=0.2)
    train2, val2, test2 = temporal_split(interactions, test_size=0.2, validation_size=0.2)

    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(val1, val2)
    pd.testing.assert_frame_equal(test1, test2)


def test_temporal_split_logs_cold_start_user_count(caplog):
    interactions = _make_interactions(100)

    with caplog.at_level("INFO"):
        temporal_split(interactions, test_size=0.2, validation_size=0.2)

    assert "cold-start" in caplog.text
