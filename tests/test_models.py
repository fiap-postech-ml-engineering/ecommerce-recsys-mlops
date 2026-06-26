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
