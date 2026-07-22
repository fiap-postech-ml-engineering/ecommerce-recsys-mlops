import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.models.dataset import InteractionsDataset


@pytest.fixture
def sparse_interactions() -> pd.DataFrame:
    # IDs propositalmente esparsos (com buracos), pra validar o mapeamento contíguo
    return pd.DataFrame(
        {
            "user_id": [100, 100, 205, 999],
            "item_id": [5000, 8123, 5000, 12],
            "score": [3, 1, 2, 1],
        }
    )


def test_len_matches_number_of_interactions(sparse_interactions):
    dataset = InteractionsDataset(sparse_interactions)
    assert len(dataset) == 4


def test_ids_mapped_to_contiguous_indices(sparse_interactions):
    dataset = InteractionsDataset(sparse_interactions)
    assert dataset.n_users == 3
    assert dataset.n_items == 3
    assert set(dataset.user_id_to_idx.values()) == {0, 1, 2}
    assert set(dataset.item_id_to_idx.values()) == {0, 1, 2}


def test_getitem_returns_rating_correctly(sparse_interactions):
    dataset = InteractionsDataset(sparse_interactions)
    user_idx, item_idx, rating = dataset[0]
    assert rating.item() == 3.0
    assert user_idx.item() == dataset.user_id_to_idx[100]
    assert item_idx.item() == dataset.item_id_to_idx[5000]


def test_smoke_dataloader_batch_shapes(sparse_interactions):
    dataset = InteractionsDataset(sparse_interactions)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    users, items, ratings = next(iter(loader))

    assert users.shape == (2,)
    assert items.shape == (2,)
    assert ratings.shape == (2,)
    assert users.dtype == torch.int64
    assert ratings.dtype == torch.float32