"""Smoke test end-to-end do pipeline DVC (TCF1-129).

Roda a cadeia preprocess -> train -> evaluate como chamadas de função Python
diretas, sobre uma amostra sintética pequena — sem invocar o binário `dvc` nem
depender do dataset real (RetailRocket via kagglehub).
"""

import pandas as pd
import pytest

from src.config import get_settings
from src.data.preprocessor import build_interactions
from src.evaluation.evaluate import evaluate_model
from src.models.factory import RecommenderFactory


def _synthetic_events(n_users: int = 20, n_items: int = 10) -> pd.DataFrame:
    """Gera eventos sintéticos com volume suficiente para sobreviver ao k-core filter."""
    rows = []
    base_timestamp = pd.Timestamp("2026-01-01")
    for user_id in range(n_users):
        for item_id in range(n_items):
            if (user_id + item_id) % 2 == 0:
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "event": "transaction",
                    "value": 10.0 + item_id,
                    "timestamp": base_timestamp + pd.Timedelta(days=user_id, hours=item_id),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.integration
def test_dvc_pipeline_smoke_end_to_end():
    settings = get_settings()
    events = _synthetic_events()

    interactions = build_interactions(events)
    assert not interactions.empty

    model = RecommenderFactory.create("itemknn", {})
    model.fit(interactions)

    metrics = evaluate_model(model, interactions, interactions, k=settings.RECOMMENDATION_K)

    expected_keys = {"precision", "recall", "ndcg", "hit_rate", "coverage", "revenue"}
    assert set(metrics.keys()) == expected_keys
    assert all(isinstance(value, float) for value in metrics.values())
