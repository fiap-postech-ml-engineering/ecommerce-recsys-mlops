import numpy as np
import pandas as pd
import pytest

from src.config import get_settings
from src.models.als import ALSRecommender
from src.models.base import BaseRecommender
from src.models.bpr import BPRRecommender
from src.models.factory import RecommenderFactory
from src.models.itemknn import ItemKNNRecommender
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
    [
        PopularityRecommender,
        SVDRecommender,
        MLPRecommender,
        ALSRecommender,
        BPRRecommender,
        ItemKNNRecommender,
    ],
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
        ("als", ALSRecommender),
        ("bpr", BPRRecommender),
        ("itemknn", ItemKNNRecommender),
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


def _make_implicit_interactions() -> pd.DataFrame:
    """Catálogo maior que `_make_svd_interactions()`: os modelos `implicit` (ALS/BPR/

    ItemKNN) não têm fallback de viés para cold-start (retornam lista vazia) — o usuário
    de teste precisa ter itens não vistos sobrando no catálogo pra `recommend()` ter o
    que retornar, o que o fixture do SVD (3 itens, usuário 1 vê todos) não garante.
    """
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
            "item_id": [10, 20, 10, 30, 40, 20, 40, 50, 30, 60],
            "score": [3, 1, 2, 3, 1, 1, 2, 3, 2, 1],
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


@pytest.mark.unit
@pytest.mark.model
def test_als_recommender_recommend_excludes_items_seen_in_training():
    interactions = _make_implicit_interactions()
    model = ALSRecommender({"factors": 2, "iterations": 5})
    model.fit(interactions)

    recs = model.recommend(user_id=1, k=5)

    assert set(recs).isdisjoint({10, 20})


@pytest.mark.unit
@pytest.mark.model
def test_als_recommender_recommend_respects_k():
    interactions = _make_implicit_interactions()
    model = ALSRecommender({"factors": 2, "iterations": 5})
    model.fit(interactions)

    assert len(model.recommend(user_id=1, k=2)) == 2


@pytest.mark.unit
@pytest.mark.model
def test_als_recommender_get_params_reflects_config():
    model = ALSRecommender(
        {"factors": 7, "regularization": 0.1, "alpha": 20.0, "iterations": 3, "random_state": 7}
    )

    assert model.get_params() == {
        "model": "als",
        "factors": 7,
        "regularization": 0.1,
        "alpha": 20.0,
        "iterations": 3,
        "random_state": 7,
    }


@pytest.mark.unit
@pytest.mark.model
def test_als_recommender_uses_settings_defaults_when_config_omits_values():
    settings = get_settings()
    model = ALSRecommender({})

    params = model.get_params()

    assert params["factors"] == settings.ALS_FACTORS
    assert params["regularization"] == settings.ALS_REGULARIZATION
    assert params["alpha"] == settings.ALS_ALPHA
    assert params["iterations"] == settings.ALS_ITERATIONS
    assert params["random_state"] == settings.RANDOM_SEED


@pytest.mark.unit
@pytest.mark.model
def test_als_recommender_recommend_raises_before_fit():
    model = ALSRecommender({"factors": 2, "iterations": 5})

    with pytest.raises(RuntimeError):
        model.recommend(user_id=1, k=2)


@pytest.mark.unit
@pytest.mark.model
def test_als_recommender_recommend_returns_empty_list_for_unknown_user():
    interactions = _make_implicit_interactions()
    model = ALSRecommender({"factors": 2, "iterations": 5})
    model.fit(interactions)

    assert model.recommend(user_id=999, k=5) == []


@pytest.mark.unit
@pytest.mark.model
def test_bpr_recommender_recommend_excludes_items_seen_in_training():
    interactions = _make_implicit_interactions()
    model = BPRRecommender({"factors": 2, "iterations": 10})
    model.fit(interactions)

    recs = model.recommend(user_id=1, k=5)

    assert set(recs).isdisjoint({10, 20})


@pytest.mark.unit
@pytest.mark.model
def test_bpr_recommender_recommend_respects_k():
    interactions = _make_implicit_interactions()
    model = BPRRecommender({"factors": 2, "iterations": 10})
    model.fit(interactions)

    assert len(model.recommend(user_id=1, k=2)) == 2


@pytest.mark.unit
@pytest.mark.model
def test_bpr_recommender_get_params_reflects_config():
    model = BPRRecommender(
        {
            "factors": 7,
            "learning_rate": 0.1,
            "regularization": 0.1,
            "iterations": 3,
            "random_state": 7,
        }
    )

    assert model.get_params() == {
        "model": "bpr",
        "factors": 7,
        "learning_rate": 0.1,
        "regularization": 0.1,
        "iterations": 3,
        "random_state": 7,
    }


@pytest.mark.unit
@pytest.mark.model
def test_bpr_recommender_uses_settings_defaults_when_config_omits_values():
    settings = get_settings()
    model = BPRRecommender({})

    params = model.get_params()

    assert params["factors"] == settings.BPR_FACTORS
    assert params["learning_rate"] == settings.BPR_LEARNING_RATE
    assert params["regularization"] == settings.BPR_REGULARIZATION
    assert params["iterations"] == settings.BPR_ITERATIONS
    assert params["random_state"] == settings.RANDOM_SEED


@pytest.mark.unit
@pytest.mark.model
def test_bpr_recommender_recommend_raises_before_fit():
    model = BPRRecommender({"factors": 2, "iterations": 10})

    with pytest.raises(RuntimeError):
        model.recommend(user_id=1, k=2)


@pytest.mark.unit
@pytest.mark.model
def test_bpr_recommender_recommend_returns_empty_list_for_unknown_user():
    interactions = _make_implicit_interactions()
    model = BPRRecommender({"factors": 2, "iterations": 10})
    model.fit(interactions)

    assert model.recommend(user_id=999, k=5) == []


@pytest.mark.unit
@pytest.mark.model
def test_itemknn_recommender_recommend_excludes_items_seen_in_training():
    interactions = _make_implicit_interactions()
    model = ItemKNNRecommender({"n_neighbors": 4})
    model.fit(interactions)

    recs = model.recommend(user_id=1, k=5)

    assert set(recs).isdisjoint({10, 20})


@pytest.mark.unit
@pytest.mark.model
def test_itemknn_recommender_recommend_respects_k():
    interactions = _make_implicit_interactions()
    model = ItemKNNRecommender({"n_neighbors": 4})
    model.fit(interactions)

    assert len(model.recommend(user_id=1, k=2)) == 2


@pytest.mark.unit
@pytest.mark.model
def test_itemknn_recommender_get_params_reflects_config():
    model = ItemKNNRecommender({"n_neighbors": 5})

    assert model.get_params() == {"model": "itemknn", "n_neighbors": 5}


@pytest.mark.unit
@pytest.mark.model
def test_itemknn_recommender_uses_settings_defaults_when_config_omits_values():
    settings = get_settings()
    model = ItemKNNRecommender({})

    assert model.get_params()["n_neighbors"] == settings.ITEMKNN_N_NEIGHBORS


@pytest.mark.unit
@pytest.mark.model
def test_itemknn_recommender_recommend_raises_before_fit():
    model = ItemKNNRecommender({"n_neighbors": 2})

    with pytest.raises(RuntimeError):
        model.recommend(user_id=1, k=2)


@pytest.mark.unit
@pytest.mark.model
def test_itemknn_recommender_recommend_returns_empty_list_for_unknown_user():
    interactions = _make_implicit_interactions()
    model = ItemKNNRecommender({"n_neighbors": 2})
    model.fit(interactions)

    assert model.recommend(user_id=999, k=5) == []


def _mlp_test_config() -> dict:
    """Config minúscula pra manter os testes de MLP rápidos (sem GPU/época longa)."""
    return {
        "embedding_dim": 4,
        "hidden_dims": [8],
        "epochs": 2,
        "batch_size": 4,
        "negative_samples": 1,
        "random_state": 42,
    }


@pytest.mark.unit
@pytest.mark.model
def test_mlp_recommender_recommend_raises_before_fit():
    model = MLPRecommender(_mlp_test_config())

    with pytest.raises(RuntimeError):
        model.recommend(user_id=1, k=2)


@pytest.mark.unit
@pytest.mark.model
def test_mlp_recommender_recommend_returns_empty_list_for_unknown_user():
    interactions = _make_implicit_interactions()
    model = MLPRecommender(_mlp_test_config())
    model.fit(interactions)

    assert model.recommend(user_id=999, k=5) == []


@pytest.mark.unit
@pytest.mark.model
def test_mlp_recommender_recommend_excludes_items_seen_in_training():
    interactions = _make_implicit_interactions()
    model = MLPRecommender(_mlp_test_config())
    model.fit(interactions)

    recs = model.recommend(user_id=1, k=5)

    assert set(recs).isdisjoint({10, 20})


@pytest.mark.unit
@pytest.mark.model
def test_mlp_recommender_recommend_respects_k():
    interactions = _make_implicit_interactions()
    model = MLPRecommender(_mlp_test_config())
    model.fit(interactions)

    assert len(model.recommend(user_id=1, k=2)) == 2


@pytest.mark.unit
@pytest.mark.model
def test_mlp_recommender_get_params_reflects_config():
    config = {
        "embedding_dim": 8,
        "hidden_dims": [16, 8],
        "epochs": 3,
        "learning_rate": 0.01,
        "batch_size": 16,
        "negative_samples": 2,
        "early_stopping_patience": 1,
        "random_state": 7,
    }
    model = MLPRecommender(config)

    assert model.get_params() == {
        "model": "mlp",
        "embedding_dim": 8,
        "hidden_dims": [16, 8],
        "epochs": 3,
        "epochs_trained": None,
        "learning_rate": 0.01,
        "batch_size": 16,
        "negative_samples": 2,
        "early_stopping_patience": 1,
        "random_state": 7,
    }


@pytest.mark.unit
@pytest.mark.model
def test_mlp_recommender_uses_settings_defaults_when_config_omits_values():
    settings = get_settings()
    model = MLPRecommender({})

    params = model.get_params()

    assert params["embedding_dim"] == settings.MLP_EMBEDDING_DIM
    assert params["hidden_dims"] == settings.MLP_HIDDEN_DIMS
    assert params["epochs"] == settings.MLP_EPOCHS
    assert params["learning_rate"] == settings.MLP_LEARNING_RATE
    assert params["batch_size"] == settings.MLP_BATCH_SIZE
    assert params["negative_samples"] == settings.MLP_NEGATIVE_SAMPLES
    assert params["early_stopping_patience"] == settings.MLP_EARLY_STOPPING_PATIENCE
    assert params["random_state"] == settings.RANDOM_SEED


@pytest.mark.unit
@pytest.mark.model
def test_mlp_recommender_training_history_has_train_and_val_loss_per_epoch():
    interactions = _make_implicit_interactions()
    config = _mlp_test_config()
    model = MLPRecommender(config)

    model.fit(interactions)

    assert len(model.training_history["train_loss"]) == config["epochs"]
    assert len(model.training_history["val_loss"]) == config["epochs"]


@pytest.mark.unit
@pytest.mark.model
def test_mlp_recommender_fit_is_reproducible_with_same_seed():
    interactions = _make_implicit_interactions()
    model_a = MLPRecommender(_mlp_test_config())
    model_b = MLPRecommender(_mlp_test_config())

    model_a.fit(interactions)
    model_b.fit(interactions)

    assert model_a.recommend(user_id=1, k=5) == model_b.recommend(user_id=1, k=5)


@pytest.mark.unit
@pytest.mark.model
def test_mlp_recommender_epochs_trained_is_none_before_fit():
    model = MLPRecommender(_mlp_test_config())

    assert model.epochs_trained is None


@pytest.mark.unit
@pytest.mark.model
def test_mlp_recommender_epochs_trained_reflects_actual_epochs_after_fit():
    interactions = _make_implicit_interactions()
    config = _mlp_test_config()
    model = MLPRecommender(config)

    model.fit(interactions)

    assert 1 <= model.epochs_trained <= config["epochs"]


@pytest.mark.unit
@pytest.mark.model
def test_mlp_recommender_early_stopping_halts_before_configured_max_epochs(monkeypatch):
    interactions = _make_implicit_interactions()
    config = _mlp_test_config() | {"epochs": 10, "early_stopping_patience": 1}
    model = MLPRecommender(config)

    # val_loss piora a cada época após a 1ª — patience=1 deve interromper cedo.
    worsening_val_losses = iter([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    monkeypatch.setattr(
        model, "_evaluate_loss", lambda *_args, **_kwargs: next(worsening_val_losses)
    )

    model.fit(interactions)

    assert model.epochs_trained == 2
    assert model.get_params()["epochs_trained"] == 2
