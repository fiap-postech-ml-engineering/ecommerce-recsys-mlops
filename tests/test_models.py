import numpy as np
import pandas as pd
import pytest

from src.models.base import BaseRecommender
from src.models.factory import RecommenderFactory
from src.models.mlp import MLPRecommender
from src.models.popularity import PopularityRecommender
from src.models.svd import SVDRecommender


class _DummyRecommender(BaseRecommender):
    def fit(self, interactions: pd.DataFrame) -> None:
        self.fitted = True

    def recommend(self, user_id: int, k: int) -> list[int]:
        return list(range(k))

    def get_params(self) -> dict:
        return {}


@pytest.mark.unit
@pytest.mark.model
def test_base_recommender_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseRecommender()


@pytest.mark.unit
@pytest.mark.model
def test_concrete_subclass_implements_strategy_interface():
    model = _DummyRecommender()

    model.fit(pd.DataFrame())

    assert model.recommend(user_id=1, k=3) == [0, 1, 2]
    assert model.get_params() == {}


@pytest.mark.unit
@pytest.mark.model
@pytest.mark.parametrize(
    "recommender_cls",
    [PopularityRecommender, SVDRecommender, MLPRecommender],
)
def test_baseline_stubs_are_declared_as_base_recommender_subclasses(recommender_cls):
    assert issubclass(recommender_cls, BaseRecommender)


@pytest.mark.unit
@pytest.mark.model
@pytest.mark.parametrize(
    ("name", "expected_cls"),
    [
        ("popularity", PopularityRecommender),
        ("svd", SVDRecommender),
        ("mlp", MLPRecommender),
    ],
)
def test_factory_creates_instance_of_correct_class(name, expected_cls):
    model = RecommenderFactory.create(name, {})

    assert isinstance(model, expected_cls)


@pytest.mark.unit
@pytest.mark.model
def test_factory_raises_value_error_on_unknown_model():
    with pytest.raises(ValueError, match="Modelo desconhecido"):
        RecommenderFactory.create("invalido", {})


@pytest.mark.unit
@pytest.mark.model
def test_factory_passes_config_to_constructor():
    config = {"n_factors": 50}

    model = RecommenderFactory.create("svd", config)

    assert model.config == config


def _make_interactions(scores_by_item: dict[int, int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": list(range(len(scores_by_item))),
            "item_id": list(scores_by_item.keys()),
            "score": list(scores_by_item.values()),
        }
    )


@pytest.mark.unit
@pytest.mark.model
def test_popularity_recommender_fit_ranks_items_by_score_descending():
    interactions = _make_interactions({10: 1, 20: 5, 30: 3})
    model = PopularityRecommender()

    model.fit(interactions)

    assert model.recommend(user_id=1, k=3) == [20, 30, 10]


@pytest.mark.unit
@pytest.mark.model
def test_popularity_recommender_recommend_ignores_user_id():
    interactions = _make_interactions({10: 1, 20: 5, 30: 3})
    model = PopularityRecommender()
    model.fit(interactions)

    assert model.recommend(user_id=1, k=2) == model.recommend(user_id=999, k=2)


@pytest.mark.unit
@pytest.mark.model
def test_popularity_recommender_recommend_respects_k():
    interactions = _make_interactions({10: 1, 20: 5, 30: 3})
    model = PopularityRecommender()
    model.fit(interactions)

    assert len(model.recommend(user_id=1, k=2)) == 2


@pytest.mark.unit
@pytest.mark.model
def test_popularity_recommender_get_params():
    model = PopularityRecommender()

    assert model.get_params() == {"model": "popularity"}


def _make_svd_interactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 1],
            "item_id": [10, 20, 10, 30, 20, 30, 30],
            "score": [3, 1, 2, 3, 1, 2, 1],
        }
    )


@pytest.mark.unit
@pytest.mark.model
def test_svd_recommender_recommend_excludes_items_seen_in_training():
    interactions = _make_svd_interactions()
    model = SVDRecommender({"n_factors": 2, "n_epochs": 2})
    model.fit(interactions)

    recs = model.recommend(user_id=1, k=5)

    assert set(recs).isdisjoint({10, 20, 30})


@pytest.mark.unit
@pytest.mark.model
def test_svd_recommender_recommend_respects_k():
    interactions = _make_svd_interactions()
    model = SVDRecommender({"n_factors": 2, "n_epochs": 2})
    model.fit(interactions)

    assert len(model.recommend(user_id=999, k=2)) == 2


@pytest.mark.unit
@pytest.mark.model
def test_svd_recommender_get_params_reflects_config():
    model = SVDRecommender(
        {"n_factors": 7, "n_epochs": 3, "lr_all": 0.02, "reg_all": 0.1, "random_state": 7}
    )

    assert model.get_params() == {
        "model": "svd",
        "n_factors": 7,
        "n_epochs": 3,
        "lr_all": 0.02,
        "reg_all": 0.1,
        "random_state": 7,
    }


@pytest.mark.unit
@pytest.mark.model
def test_svd_recommender_uses_settings_defaults_when_config_omits_values():
    model = SVDRecommender({})

    params = model.get_params()

    assert params["n_factors"] > 0
    assert params["n_epochs"] > 0
    assert params["lr_all"] == 0.005
    assert params["reg_all"] == 0.02


@pytest.mark.unit
@pytest.mark.model
def test_svd_recommender_passes_lr_all_and_reg_all_to_algo():
    interactions = _make_svd_interactions()
    model = SVDRecommender({"n_factors": 2, "n_epochs": 2, "lr_all": 0.02, "reg_all": 0.1})
    model.fit(interactions)

    assert model._algo.lr_bu == 0.02
    assert model._algo.reg_bu == 0.1


@pytest.mark.unit
@pytest.mark.model
def test_svd_recommender_recommend_raises_before_fit():
    model = SVDRecommender({"n_factors": 2, "n_epochs": 2})

    with pytest.raises(RuntimeError):
        model.recommend(user_id=1, k=2)


@pytest.mark.unit
@pytest.mark.model
def test_svd_recommender_scores_known_user_with_personalized_embeddings():
    """Usuário presente no treino deve usar o inner_uid correto (não o raw user_id)."""
    interactions = _make_svd_interactions()
    model = SVDRecommender({"n_factors": 2, "n_epochs": 5})
    model.fit(interactions)

    known_user_scores = model._score_all_items(user_id=1)
    cold_start_scores = model._trainset.global_mean + model._algo.bi

    assert not np.allclose(known_user_scores, cold_start_scores)
