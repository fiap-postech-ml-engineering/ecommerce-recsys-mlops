import pandas as pd

from src.data.preprocessor import (
    BasePreprocessor,
    WeightedInteractionPreprocessor,
    build_interactions,
    build_preprocessing_pipeline,
)


def _make_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "item_id": [10, 10, 20, 10],
            "event": ["view", "addtocart", "view", "transaction"],
            "value": [50.0, 50.0, 80.0, 50.0],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-02", "2020-01-04"]),
        }
    )


def test_build_interactions_aggregates_score_by_user_and_item():
    result = (
        build_interactions(_make_events())
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )

    assert list(result.columns) == ["user_id", "item_id", "score", "value", "timestamp"]
    assert len(result) == 3

    user1_item10 = result[(result["user_id"] == 1) & (result["item_id"] == 10)].iloc[0]
    assert user1_item10["score"] == 3  # view (1) + addtocart (2)
    assert user1_item10["value"] == 50.0
    assert user1_item10["timestamp"] == pd.Timestamp("2020-01-03")

    user1_item20 = result[(result["user_id"] == 1) & (result["item_id"] == 20)].iloc[0]
    assert user1_item20["score"] == 1  # view

    user2_item10 = result[(result["user_id"] == 2) & (result["item_id"] == 10)].iloc[0]
    assert user2_item10["score"] == 3  # transaction


def test_build_preprocessing_pipeline_transforms_events_into_interactions():
    events = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 10],
            "event": ["view", "transaction"],
            "value": [50.0, 50.0],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )

    pipeline = build_preprocessing_pipeline()
    result = pipeline.fit_transform(events)

    assert list(result.columns) == ["user_id", "item_id", "score", "value", "timestamp"]
    assert result.iloc[0]["score"] == 4  # view (1) + transaction (3)


def test_weighted_interaction_preprocessor_is_base_preprocessor_subclass():
    assert issubclass(WeightedInteractionPreprocessor, BasePreprocessor)


def test_weighted_interaction_preprocessor_transform_produces_correct_scores():
    result = (
        WeightedInteractionPreprocessor()
        .transform(_make_events())
        .sort_values(["user_id", "item_id"])
        .reset_index(drop=True)
    )

    assert list(result.columns) == ["user_id", "item_id", "score", "value", "timestamp"]
    user1_item10 = result[(result["user_id"] == 1) & (result["item_id"] == 10)].iloc[0]
    assert user1_item10["score"] == 3  # view (1) + addtocart (2)


def test_build_preprocessing_pipeline_accepts_custom_strategy():
    class DoubleScorePreprocessor(BasePreprocessor):
        def transform(self, events: pd.DataFrame) -> pd.DataFrame:
            result = WeightedInteractionPreprocessor().transform(events)
            result = result.copy()
            result["score"] = result["score"] * 2
            return result

    events = pd.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 10],
            "event": ["view", "transaction"],
            "value": [50.0, 50.0],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )

    pipeline = build_preprocessing_pipeline(preprocessor=DoubleScorePreprocessor())
    result = pipeline.fit_transform(events)

    assert result.iloc[0]["score"] == 8  # (view 1 + transaction 3) * 2
