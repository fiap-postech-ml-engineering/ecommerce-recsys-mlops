"""Script de avaliação final — ponto de entrada do stage "evaluate" do DVC.

Carrega o modelo de produção persistido por `src/training/train.py`, avalia sobre
`test_df` (tocado uma única vez, aqui) e salva métricas em `metrics.json` e um
gráfico em `plots/`.
"""

import json
import logging
from pathlib import Path
import statistics

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.config import BASE_DIR, MODELS_DIR, get_settings
from src.data.filtering import k_core_filter
from src.data.preprocessor import EVENT_WEIGHTS
from src.data.split import temporal_split
from src.metrics import (
    coverage,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    revenue_at_k,
)
from src.models.base import BaseRecommender

logger = logging.getLogger(__name__)

INTERACTIONS_PATH = BASE_DIR / "data" / "processed" / "interactions.parquet"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = BASE_DIR / "metrics.json"
PLOTS_DIR = BASE_DIR / "plots"

# build_interactions() perde o tipo de evento na agregação (score = soma de pesos por
# par user/item). score >= peso de "transaction" aproxima "houve transação" — mesmo
# proxy usado em notebooks/02_experiments.ipynb, seção 3.
TRANSACTION_SCORE_THRESHOLD = EVENT_WEIGHTS["transaction"]


def _load_model(model_path: Path = MODEL_PATH) -> BaseRecommender:
    """Carrega o modelo de produção persistido por `src/training/train.py`.

    Args:
        model_path: Caminho para o artefato `model.joblib`.

    Returns:
        Instância treinada de `BaseRecommender`.
    """
    return joblib.load(model_path)


def _load_eval_data(
    interactions_path: Path = INTERACTIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recalcula `train_df`/`test_df` com o mesmo split e k-core do treino.

    Reaplica `temporal_split()` e `k_core_filter()` com os mesmos parâmetros de
    `Settings` usados por `src/training/train.py`, para que `eval_users` e
    `catalog_size` reflitam o universo que o modelo realmente viu no treino.
    `apply_log_scaling()` não é necessário aqui: `eval_users`/`catalog_size`
    dependem só de `user_id`/`item_id`, não da escala do score.

    Args:
        interactions_path: Caminho para o parquet de interações pré-processadas.

    Returns:
        Tupla `(train_df, test_df)` pós k-core filtering. `test_df` é usado uma
        única vez, nesta função.
    """
    settings = get_settings()
    interactions = pd.read_parquet(interactions_path)

    train_df, val_df, test_df = temporal_split(
        interactions,
        test_size=settings.TEST_SIZE,
        validation_size=settings.VALIDATION_SIZE,
    )
    train_df, _, test_df = k_core_filter(
        train_df,
        val_df,
        test_df,
        min_user_interactions=settings.MIN_USER_INTERACTIONS,
        min_item_interactions=settings.MIN_ITEM_INTERACTIONS,
    )
    return train_df, test_df


def _relevant_items_by_user(split_df: pd.DataFrame) -> dict[int, set[int]]:
    """Mapeia `user_id` aos itens com score de transação em `split_df`."""
    purchases = split_df[split_df["score"] >= TRANSACTION_SCORE_THRESHOLD]
    return purchases.groupby("user_id")["item_id"].apply(set).to_dict()


def _value_by_item(split_df: pd.DataFrame) -> dict[int, float]:
    """Mapeia `item_id` ao valor monetário mais recente em `split_df`."""
    return split_df.drop_duplicates("item_id").set_index("item_id")["value"].to_dict()


def _safe_mean(values: list[float]) -> float:
    """Média de `values`, ou 0.0 se a lista estiver vazia (nenhum eval_user com compra)."""
    return statistics.mean(values) if values else 0.0


def _score_user(
    model: BaseRecommender,
    user_id: int,
    relevant: set[int],
    value_by_item: dict[int, float],
    k: int,
) -> tuple[list[int], dict[str, float]]:
    """Gera recomendações e calcula as métricas de ranking/receita de um usuário."""
    recs = model.recommend(user_id=user_id, k=k)
    relevant_with_value = {i: value_by_item[i] for i in relevant if i in value_by_item}
    return recs, {
        "precision": precision_at_k(recs, relevant, k),
        "recall": recall_at_k(recs, relevant, k),
        "ndcg": ndcg_at_k(recs, relevant, k),
        "hit_rate": hit_rate_at_k(recs, relevant, k),
        "revenue": revenue_at_k(recs, relevant_with_value, k),
    }


def evaluate_model(
    model: BaseRecommender,
    split_df: pd.DataFrame,
    train_df: pd.DataFrame,
    k: int,
) -> dict[str, float]:
    """Avalia um `BaseRecommender` já treinado sobre `split_df`.

    Porta `evaluate_model()` de `notebooks/02_experiments.ipynb` (seção 3):
    usuários cold-start em `split_df` (sem histórico em `train_df`) são
    excluídos; "relevante" é aproximado por score >= peso de "transaction" em
    `EVENT_WEIGHTS`. `catalog_size` é o número de itens distintos em `train_df`
    (universo pós k-core que o modelo viu no treino).

    Args:
        model: Modelo já treinado, interface `BaseRecommender`.
        split_df: Split a avaliar (`test_df` ou `val_df`) — `test_df` nunca deve
            ser reutilizado além desta chamada.
        train_df: Split de treino, usado para restringir `eval_users` e calcular
            `catalog_size`.
        k: Corte de avaliação (`Settings.RECOMMENDATION_K`).

    Returns:
        Dict com as 6 métricas: precision, recall, ndcg, hit_rate, coverage, revenue.
    """
    eval_users = set(split_df["user_id"]) & set(train_df["user_id"])
    relevant_by_user = _relevant_items_by_user(split_df)
    value_by_item = _value_by_item(split_df)

    scores: dict[str, list[float]] = {
        "precision": [],
        "recall": [],
        "ndcg": [],
        "hit_rate": [],
        "revenue": [],
    }
    all_recs: list[list[int]] = []

    for user_id in eval_users:
        relevant = relevant_by_user.get(user_id, set())
        if not relevant:
            continue
        recs, user_scores = _score_user(model, user_id, relevant, value_by_item, k)
        all_recs.append(recs)
        for name, value in user_scores.items():
            scores[name].append(value)

    catalog_size = train_df["item_id"].nunique()
    return {
        "precision": _safe_mean(scores["precision"]),
        "recall": _safe_mean(scores["recall"]),
        "ndcg": _safe_mean(scores["ndcg"]),
        "hit_rate": _safe_mean(scores["hit_rate"]),
        "coverage": coverage(all_recs, catalog_size),
        "revenue": _safe_mean(scores["revenue"]),
    }


def _save_metrics(metrics: dict[str, float], metrics_path: Path = METRICS_PATH) -> None:
    """Salva as métricas em JSON plano, formato esperado por `dvc metrics show`.

    Args:
        metrics: Dict com as 6 métricas retornadas por `evaluate_model()`.
        metrics_path: Caminho de destino do JSON.
    """
    metrics_path.write_text(json.dumps(metrics, indent=2))


def _plot_metrics(metrics: dict[str, float], plots_dir: Path = PLOTS_DIR) -> Path:
    """Gera um gráfico de barras das 6 métricas e salva em `plots/`.

    Duas subplots: `revenue` fica sozinho porque sua escala (monetária) é ordens
    de grandeza maior que as demais métricas, todas em [0, 1] — um único eixo
    compartilhado deixaria essas 5 invisíveis.

    Args:
        metrics: Dict com as 6 métricas retornadas por `evaluate_model()`.
        plots_dir: Diretório de destino do gráfico.

    Returns:
        Caminho do arquivo de gráfico salvo.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "metrics.png"
    ratio_metrics = {k: v for k, v in metrics.items() if k != "revenue"}

    fig, (ax_ratio, ax_revenue) = plt.subplots(1, 2, figsize=(11, 5), width_ratios=[3, 1])
    ax_ratio.bar(list(ratio_metrics.keys()), list(ratio_metrics.values()), color="steelblue")
    ax_ratio.set_ylabel("Valor")
    ax_ratio.set_ylim(0, 1)
    ax_ratio.set_title("Precision / Recall / NDCG / Hit Rate / Coverage")

    ax_revenue.bar(["revenue"], [metrics["revenue"]], color="darkorange")
    ax_revenue.set_title("Revenue@K")

    fig.suptitle("Métricas de avaliação — modelo de produção")
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    return plot_path


def main() -> None:
    """Avalia o modelo de produção sobre `test_df` e salva métricas e gráfico."""
    settings = get_settings()
    model = _load_model()
    train_df, test_df = _load_eval_data()

    logger.info(
        "Avaliando modelo de produção sobre test_df: %d interações, %d usuários",
        len(test_df),
        test_df["user_id"].nunique(),
    )
    metrics = evaluate_model(model, test_df, train_df, settings.RECOMMENDATION_K)
    logger.info("Métricas: %s", metrics)

    _save_metrics(metrics)
    logger.info("Métricas salvas em: %s", METRICS_PATH)

    plot_path = _plot_metrics(metrics)
    logger.info("Gráfico salvo em: %s", plot_path)


if __name__ == "__main__":
    main()
