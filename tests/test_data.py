import pandas as pd
import pytest

from src.config import DATA_DIR
from src.data import load_dataset
from src.data.loader import (
    RAW_FILENAMES,
    _consolidate_dataset,
    _ensure_raw_dataset,
    _extract_latest_item_values,
)


def test_extract_latest_item_values_picks_latest_and_cleans_values():
    item_properties = pd.DataFrame(
        {
            "timestamp": [100, 200, 100, 100, 100],
            "itemid": [1, 1, 2, 3, 4],
            "property": ["790", "790", "790", "790", "790"],
            "value": ["n100", "n150", "n0", "n-50", "nabc"],
        }
    )

    result = _extract_latest_item_values(item_properties)

    assert result.to_dict() == {1: 150.0}


def test_consolidate_dataset_builds_expected_schema():
    events = pd.DataFrame(
        {
            "visitorid": [10, 10, 11],
            "itemid": [1, 2, 1],
            "event": ["view", "addtocart", "transaction"],
            "datetime": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        }
    )
    item_properties = pd.DataFrame(
        {
            "timestamp": [50],
            "itemid": [1],
            "property": ["790"],
            "value": ["n200"],
        }
    )

    result = _consolidate_dataset(events, item_properties)

    assert list(result.columns) == ["user_id", "item_id", "score", "value", "timestamp"]
    assert set(result["item_id"]) == {1}
    assert sorted(result["score"]) == [1, 3]
    assert (result["value"] == 200.0).all()


def test_ensure_raw_dataset_skips_download_when_files_present(tmp_path, monkeypatch):
    for name in RAW_FILENAMES:
        (tmp_path / name).touch()

    def fail_download(handle):
        raise AssertionError("dataset_download não deveria ser chamado")

    monkeypatch.setattr("src.data.loader.kagglehub.dataset_download", fail_download)

    _ensure_raw_dataset(tmp_path)


def test_ensure_raw_dataset_downloads_and_copies_missing_files(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "events.csv").write_text("present")

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    for name in RAW_FILENAMES:
        (cache_dir / name).write_text("from-cache")

    monkeypatch.setattr(
        "src.data.loader.kagglehub.dataset_download", lambda handle: str(cache_dir)
    )

    _ensure_raw_dataset(raw_dir)

    assert (raw_dir / "events.csv").read_text() == "present"
    for name in RAW_FILENAMES[1:]:
        assert (raw_dir / name).read_text() == "from-cache"


def test_load_dataset_uses_cache_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.loader.DATA_DIR", tmp_path)
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    expected = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [2],
            "score": [1],
            "value": [10.0],
            "timestamp": pd.to_datetime(["2020-01-01"]),
        }
    )
    expected.to_csv(processed_dir / "dataset_consolidated.csv", index=False)

    def fail_ensure(raw_dir):
        raise AssertionError("_ensure_raw_dataset não deveria ser chamado")

    monkeypatch.setattr("src.data.loader._ensure_raw_dataset", fail_ensure)

    result = load_dataset()

    pd.testing.assert_frame_equal(result, expected)


def test_load_dataset_builds_and_caches_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.loader.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.data.loader._ensure_raw_dataset", lambda raw_dir: None)
    monkeypatch.setattr(
        "src.data.loader._load_raw_tables", lambda raw_dir: (pd.DataFrame(), pd.DataFrame())
    )
    expected = pd.DataFrame(
        {
            "user_id": [1],
            "item_id": [2],
            "score": [3],
            "value": [10.0],
            "timestamp": pd.to_datetime(["2020-01-01"]),
        }
    )
    monkeypatch.setattr(
        "src.data.loader._consolidate_dataset", lambda events, item_properties: expected
    )

    result = load_dataset()

    pd.testing.assert_frame_equal(result, expected)
    assert (tmp_path / "processed" / "dataset_consolidated.csv").exists()


@pytest.mark.integration
@pytest.mark.slow
def test_load_dataset_against_real_raw_files():
    raw_dir = DATA_DIR / "raw"
    if any(not (raw_dir / f).exists() for f in RAW_FILENAMES):
        pytest.skip("Arquivos raw do RetailRocket não encontrados localmente")
    dataset = load_dataset(force_rebuild=True)
    assert set(dataset.columns) == {"user_id", "item_id", "score", "value", "timestamp"}
    assert dataset["score"].isin([1, 2, 3]).all()
