# MLflow — Boas Práticas de Experimentos (ECOMM_RECSYS)

Este documento define como organizar experimentos, runs, tags, métricas e
artefatos do MLflow neste projeto. Para configurar credenciais e validar a
conexão com o Databricks, veja `docs/internal/configuracao_mlflow.md` — este
guia é sobre **convenções de uso**, não sobre setup.

As métricas e hiperparâmetros citados aqui vêm de `docs/internal/planejamento_inicial.md`
e de `src/config.py` (`Settings`) — não duplique valores, sempre leia de lá.

-----

## 1. Experimentos

Use três experimentos lógicos, escolhidos via
`configure_mlflow_tracking(experiment_name=...)`:

```
02 - ECOMM_RECSYS - notebook_baselines_training
02 - ECOMM_RECSYS - notebook_mlp_training
02 - ECOMM_RECSYS - Production
```

| Experimento | Quando usar |
| --- | --- |
| `notebook_baselines_training` | Exploração em `notebooks/02_experiments.ipynb` dos baselines (Popularity, SVD) |
| `notebook_mlp_training` | Exploração em `notebooks/02_experiments.ipynb` do MLP (arquitetura, hiperparâmetros) |
| `02 - ECOMM_RECSYS - Production` (default de `Settings.MLFLOW_EXPERIMENT_NAME`) | Runs "oficiais" disparadas por `src/training/train.py` / pipeline DVC: baseline, tuning, final |

`configure_mlflow_tracking()` sem argumentos usa o experimento de produção. Para
os notebooks de exploração, passe o nome explicitamente:

```python
configure_mlflow_tracking(experiment_name="02 - ECOMM_RECSYS - notebook_mlp_training")
```

-----

## 2. Tags de cada run

Use `build_experiment_tags()` (`src/tracking/mlflow_utils.py`) para montar as
tags de toda run:

```python
from src.tracking import build_experiment_tags

tags = build_experiment_tags(
    model_type="mlp",            # "popularity", "svd" ou "mlp" (chaves do RecommenderFactory)
    phase="dev",                  # "dev", "baseline", "tuning" ou "final"
    dataset_name="retailrocket",
    dataset_dvc_version="a1b2c3d",  # opcional — "unknown" enquanto o DVC pipeline não existir
    extra_tags={"run_group": "notebook_mlp_dev"},  # opcional, ver seção 3
)
```

Tags retornadas: `model_type`, `phase`, `dataset_name`, `dataset_dvc_version`,
`author` (via `git config user.name`), mais qualquer `extra_tags`.

> Tags como commit/branch/host/timestamp **não são adicionadas manualmente** —
> o MLflow já registra `mlflow.source.git.commit` e `run.info.start_time`
> nativamente para cada run.

### Descrição da run

Use a tag especial `mlflow.note.content` (aparece formatada na UI):

```python
mlflow.set_tag("mlflow.note.content", "MLP com embedding_dim=64, testando hidden_dims=[128, 64].")
```

-----

## 3. Agrupamento de runs de desenvolvimento (`run_group`)

**Problema**: várias pessoas rodando o mesmo notebook geram runs soltas,
poluindo o experimento.

**Solução**: toda run de desenvolvimento recebe uma tag `run_group` via
`extra_tags`:

```python
tags = build_experiment_tags(
    model_type="mlp",
    phase="dev",
    dataset_name="retailrocket",
    extra_tags={"run_group": "notebook_mlp_dev"},
)
```

Na UI do MLflow, filtre/agrupe com:

```
tags.run_group = 'notebook_mlp_dev'
```

Runs "oficiais" (`baseline`, `tuning`, `final`) não precisam de `run_group`.

-----

## 4. Nomenclatura de runs

Padrão para runs oficiais:

```
<modelo>_<fase>_<timestamp>
```

Exemplos alinhados ao `RecommenderFactory` (`popularity`, `svd`, `mlp`):

- `popularity_baseline_20260615T143200Z`
- `svd_baseline_20260615T144500Z`
- `mlp_tuning_20260615T150030Z`
- `mlp_final_20260615T160000Z`

O timestamp no nome resolve unicidade e dá ordenação cronológica — não é
necessário repeti-lo como tag, já existe em `run.info.start_time`.

-----

## 5. Modelo, signature e Model Registry

Diretrizes gerais (os detalhes exatos por modelo são definidos quando
`src/models/` for implementado, seguindo o Strategy/Factory):

- **Sempre que possível, logue o modelo com `signature`** via
  `infer_signature`. Para o MLP (PyTorch), a signature é inferida a partir do
  forward de scoring (`(user_idx, item_idx) -> score`). Para o SVD
  (`scikit-surprise`), use o wrapper apropriado ou `mlflow.pyfunc`. Para o
  Popularity, que não tem um "modelo" no sentido clássico (apenas um ranking
  de itens), logue o artefato de ranking (`mlflow.log_artifact` /
  `mlflow.log_dict`) — signature não se aplica.
- **Run vs. versão registrada**: toda run que treina um modelo loga o artefato
  (`log_model`/`log_artifact`), mas só a run "campeã" (fase `final`) chama
  `mlflow.register_model(...)`, criando uma nova versão de
  `ecomm-recsys-<model_type>` no Model Registry.
- Promoção para `staging`/`production` é feita via aliases no Model Registry,
  manualmente após avaliação.

-----

## 6. Métricas — convenção de prefixos

Use os prefixos `model.*` (métricas de modelo) e `business.*` (métricas de
negócio), com os nomes definidos em `docs/internal/planejamento_inicial.md`:

```python
mlflow.log_metrics({
    # Métricas do modelo (Precision@K, Recall@K, NDCG@K, Hit Rate@K)
    "model.precision_at_k": 0.18,
    "model.recall_at_k": 0.12,
    "model.ndcg_at_k": 0.21,
    "model.hit_rate_at_k": 0.35,

    # Apenas para o MLP, por época/checkpoint
    "model.train_loss": 0.034,
    "model.val_loss": 0.041,

    # Métricas de negócio
    "business.coverage": 0.42,
    "business.revenue_at_k": 1530.75,
})
```

> Nomes usam `_at_k` (não `@k`) porque `@` não é um caractere válido em nomes
> de métricas/tags do MLflow. O valor de `K` é `Settings.RECOMMENDATION_K`,
> logado como parâmetro (`recommendation_k`), não precisa repetir no nome da
> métrica.

A UI do MLflow agrupa visualmente por prefixo ao comparar runs.

-----

## 7. Artefatos — estrutura padrão

| Conteúdo | `artifact_path` | Frequência |
| --- | --- | --- |
| Modelo + signature (quando aplicável) | `model` | toda run que treina um modelo |
| Mapeamento user/item ↔ índice de embedding | `data/id_mappings.json` | toda run, ou quando o dataset mudar |
| Métricas consolidadas (`model.*` + `business.*`) | `metrics.json` (raiz) | toda run |
| Curva de treino (loss por época) — apenas MLP | `diagnostics/training_curve.png` | runs de MLP |
| Comparação entre modelos (baselines vs. MLP) | `diagnostics/metrics_comparison.png` | runs `final` |
| Model card | `model_card` | apenas runs `final` registradas |

```python
with mlflow.start_run(run_name=run_name, tags=tags) as run:
    mlflow.set_tag("mlflow.note.content", run_description)

    mlflow.log_params(model.get_params())
    mlflow.log_metrics(model_metrics | business_metrics)

    mlflow.log_dict(model_metrics | business_metrics, "metrics.json")
    mlflow.log_dict(id_mappings, "data/id_mappings.json")

    if hasattr(model, "training_history"):
        mlflow.log_artifact("training_curve.png", artifact_path="diagnostics")

    if is_final_run:
        mlflow.register_model(model_uri, f"ecomm-recsys-{model_type}")
        mlflow.log_artifact("docs/model-card.md", artifact_path="model_card")
```

-----

## 8. Checklist por run

- [ ] Experimento correto (`notebook_baselines_training` / `notebook_mlp_training`
      para runs de desenvolvimento, `02 - ECOMM_RECSYS - Production` para runs oficiais)
- [ ] Tags via `build_experiment_tags(model_type, phase, dataset_name, dataset_dvc_version, ...)`
- [ ] `run_group` definido em `extra_tags` se for run de desenvolvimento
- [ ] `run_name` seguindo `<modelo>_<fase>_<timestamp>` em runs oficiais
- [ ] `mlflow.note.content` com descrição da run
- [ ] Hiperparâmetros logados via `mlflow.log_params(model.get_params())`
- [ ] Métricas com prefixo `model.*` (Precision/Recall/NDCG/Hit Rate@K) e
      `business.*` (Coverage, Revenue@K) — mínimo 4 métricas para comparar com o MLP
- [ ] Modelo logado com `signature` quando aplicável
- [ ] `data/id_mappings.json` e `metrics.json` logados
- [ ] Se for run `final`: `register_model()` + model card

-----

## 9. Exemplo completo — `PopularityRecommender` (ilustrativo)

> **Atenção**: `src/models/` ainda não existe (stub vazio). O exemplo abaixo é
> o "alvo" de como uma run de baseline vai ficar quando `PopularityRecommender`
> e `RecommenderFactory` forem implementados — serve para visualizar o fluxo
> completo, não é executável hoje.
>
> Para validar a infraestrutura de tracking (Databricks + tags) **hoje**, sem
> depender dos modelos, existe `tests/test_tracking.py::test_full_dummy_run_logs_to_mlflow_with_expected_tags`
> (`@pytest.mark.integration @pytest.mark.slow`, roda via `make test-slow`).

```python
import mlflow

from src.config import get_settings
from src.models.factory import RecommenderFactory
from src.tracking import build_experiment_tags, configure_mlflow_tracking

settings = get_settings()

configure_mlflow_tracking(experiment_name="02 - ECOMM_RECSYS - notebook_baselines_training")

tags = build_experiment_tags(
    model_type="popularity",
    phase="baseline",
    dataset_name="retailrocket",
    dataset_dvc_version="a1b2c3d",
)

run_name = "popularity_baseline_20260615T143200Z"

with mlflow.start_run(run_name=run_name, tags=tags):
    mlflow.set_tag(
        "mlflow.note.content",
        "Baseline de popularidade — ranking global dos itens mais comprados.",
    )

    model = RecommenderFactory.create("popularity", config={})
    model.fit(train_interactions)

    mlflow.log_params(model.get_params())

    model_metrics = evaluate_model_metrics(model, test_interactions, k=settings.RECOMMENDATION_K)
    business_metrics = evaluate_business_metrics(model, test_interactions, k=settings.RECOMMENDATION_K)

    mlflow.log_metrics(model_metrics | business_metrics)
    mlflow.log_dict(model_metrics | business_metrics, "metrics.json")
    mlflow.log_dict(model.get_id_mappings(), "data/id_mappings.json")

    mlflow.log_dict(model.get_ranking(), artifact_file="model/ranking.json")
```

Para o MLP, a diferença principal é logar o `signature` e o artefato do modelo
via `mlflow.pytorch.log_model(model, artifact_path="model", signature=signature)`,
além de `model.train_loss`/`model.val_loss` por época e
`diagnostics/training_curve.png`.
