import pandas as pd

from src.data.filtering import k_core_filter


def _make_interactions(rows: list[tuple[int, int]]) -> pd.DataFrame:
    n = len(rows)
    return pd.DataFrame(
        {
            "user_id": [r[0] for r in rows],
            "item_id": [r[1] for r in rows],
            "score": [1] * n,
            "value": [10.0] * n,
            "timestamp": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )


def test_k_core_filter_removes_users_below_threshold():
    train_df = _make_interactions(
        [
            (1, 10),
            (1, 11),
            (1, 12),
            (1, 13),
            (1, 14),
            (2, 10),
        ]
    )
    empty = train_df.iloc[0:0]

    filtered_train, _, _ = k_core_filter(
        train_df, empty, empty, min_user_interactions=2, min_item_interactions=1
    )

    assert set(filtered_train["user_id"]) == {1}


def test_k_core_filter_removes_items_below_threshold():
    train_df = _make_interactions(
        [
            (1, 10),
            (2, 10),
            (3, 10),
            (4, 10),
            (5, 10),
            (6, 11),
        ]
    )
    empty = train_df.iloc[0:0]

    filtered_train, _, _ = k_core_filter(
        train_df, empty, empty, min_user_interactions=1, min_item_interactions=5
    )

    assert set(filtered_train["item_id"]) == {10}
    assert 6 not in set(filtered_train["user_id"])


def test_k_core_filter_converges_on_cascading_removal():
    # 1ª passada: user2 (1 interação) e item10 (1 interação) são removidos,
    # sobrevive só (1, 11). 2ª passada: com só 1 linha restante, nem user1
    # nem item11 atingem mais o limiar de 2 -> converge em 0 linhas.
    train_df = _make_interactions(
        [
            (1, 10),
            (1, 11),
            (2, 11),
        ]
    )
    empty = train_df.iloc[0:0]

    filtered_train, _, _ = k_core_filter(
        train_df, empty, empty, min_user_interactions=2, min_item_interactions=2
    )

    assert len(filtered_train) == 0


def test_k_core_filter_restricts_val_and_test_to_train_universe():
    train_df = _make_interactions([(1, 10), (1, 11), (2, 10), (2, 11)])
    val_df = _make_interactions([(1, 10), (3, 10), (1, 99)])
    test_df = _make_interactions([(2, 11), (3, 11)])

    filtered_train, filtered_val, filtered_test = k_core_filter(
        train_df, val_df, test_df, min_user_interactions=2, min_item_interactions=2
    )

    assert set(filtered_train["user_id"]) == {1, 2}
    assert len(filtered_val) == 1
    assert filtered_val.iloc[0][["user_id", "item_id"]].tolist() == [1, 10]
    assert len(filtered_test) == 1
    assert filtered_test.iloc[0][["user_id", "item_id"]].tolist() == [2, 11]


def test_k_core_filter_logs_summary(caplog):
    train_df = _make_interactions([(1, 10), (1, 11), (2, 10), (2, 11)])
    empty = train_df.iloc[0:0]

    with caplog.at_level("INFO"):
        k_core_filter(train_df, empty, empty, min_user_interactions=2, min_item_interactions=2)

    assert "k-core filtering" in caplog.text
