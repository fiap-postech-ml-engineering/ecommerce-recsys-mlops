import pandas as pd
import pandera.pandas
import pytest

from src.data.schema import validate_interactions, validate_raw_events


def _make_valid_raw_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "event": ["view", "transaction"],
            "value": [50.0, 30.0],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )


def _make_valid_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2],
            "item_id": [10, 20],
            "score": [3, 1],
            "value": [50.0, 30.0],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )


@pytest.mark.unit
def test_raw_events_schema_accepts_valid_dataframe():
    result = validate_raw_events(_make_valid_raw_events())
    assert list(result.columns) == ["user_id", "item_id", "event", "value", "timestamp"]


@pytest.mark.unit
def test_raw_events_schema_rejects_invalid_event_type():
    df = _make_valid_raw_events()
    df["event"] = ["view", "invalid_event"]
    with pytest.raises(pandera.pandas.errors.SchemaError):
        validate_raw_events(df)


@pytest.mark.unit
def test_raw_events_schema_rejects_missing_column():
    df = _make_valid_raw_events().drop(columns=["user_id"])
    with pytest.raises(pandera.pandas.errors.SchemaError):
        validate_raw_events(df)


@pytest.mark.unit
def test_interactions_schema_accepts_valid_dataframe():
    result = validate_interactions(_make_valid_interactions())
    assert list(result.columns) == ["user_id", "item_id", "score", "value", "timestamp"]


@pytest.mark.unit
def test_interactions_schema_rejects_zero_score():
    df = _make_valid_interactions()
    df["score"] = [0, 1]
    with pytest.raises(pandera.pandas.errors.SchemaError):
        validate_interactions(df)


@pytest.mark.unit
def test_interactions_schema_rejects_missing_column():
    df = _make_valid_interactions().drop(columns=["score"])
    with pytest.raises(pandera.pandas.errors.SchemaError):
        validate_interactions(df)


@pytest.mark.unit
def test_interactions_schema_accepts_nullable_value():
    df = _make_valid_interactions()
    df["value"] = [50.0, None]
    result = validate_interactions(df)
    assert result["value"].isna().any()
