# MLflow — Boas Práticas de Experimentos (ECOMM_RECSYS)

Este documento define como organizar experimentos, runs, tags, métricas e
artefatos do MLflow neste projeto. Para configurar credenciais e validar a
conexão com o Databricks, veja `docs/internal/mlflow/01_configuracao_mlflow.md` — este
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
| `02 - ECOMM_RECSYS - notebook_baselines_training` | Exploração em `notebooks/02_experiments.ipynb` dos baselines (Popularity, SVD) |
| `02 - ECOMM_RECSYS - notebook_mlp_training` | Exploração em `notebooks/02_experiments.ipynb` do MLP (arquitetura, hiperparâmetros) |
| `02 - ECOMM_RECSYS - Production` (default de `Settings.MLFLOW_EXPERIMENT_NAME`) | Runs "oficiais" disparadas por `src/training/train.py` / pipeline DVC: baseline, tuning, final |

`configure_mlflow_tracking()` sem argumentos usa o experimento de produção. Para
os notebooks de exploração, passe o nome explicitamente:

```python
configure_mlflow_tracking(experiment_name="02 - ECOMM_RECSYS - notebook_mlp_training")
```

-----

## 2. Tags de cada run

Use `build_experiment_tags()` (`src/tracking/mlflow_utils.py`) para montar as
tags de runs criadas fora dos notebooks (ex: `src/training/train.py`):

```python
from src.tracking import build_experiment_tags

tags = build_experiment_tags(
    model_type="mlp",            # "popularity", "svd" ou "mlp" (chaves do RecommenderFactory)
    phase="dev",                  # "dev", "baseline", "tuning" ou "final"
    dataset_name="retailrocket",    # fixo, não muda
    dataset_dvc_version="a1b2c3d",  # opcional — "unknown" enquanto o DVC pipeline não existir
    run_group="mlp_sem_fe",         # opcional — use start_notebook_run nos notebooks (seção 3)
)
```

Tags retornadas: `model_type`, `phase`, `dataset_name`, `dataset_dvc_version`,
`author` (via `git config user.name`), `run_group` (se fornecido).

> Tags como commit/branch/host/timestamp **não são adicionadas manualmente** —
> o MLflow já registra `mlflow.source.git.commit` e `run.info.start_time`
> nativamente para cada run.

### Descrição da run

Use a tag especial `mlflow.note.content` (aparece formatada na UI):

```python
mlflow.set_tag("mlflow.note.content", "MLP com embedding_dim=64, testando hidden_dims=[128, 64].")
```

-----

## 3. Hierarquia de runs nos notebooks (`start_notebook_run`)

Nos notebooks de exploração, use sempre `start_notebook_run` em vez de
`mlflow.start_run` diretamente. Essa função organiza as runs em três níveis
automaticamente:

```
Experimento: notebook_baselines_training
└── svd                              ← nível 1: modelo (organizacional, sem métricas)
    ├── svd_sem_fe_nf50              ← nível 2: teste (organizacional, sem métricas)
    │   ├── svd_sem_fe_nf50_20260616T143200Z  ← iteração (loga métricas)
    │   └── svd_sem_fe_nf50_20260616T150045Z  ← iteração de outra pessoa
    └── svd_com_fe_nf50              ← nível 2: outro teste
        └── svd_com_fe_nf50_20260616T160000Z
```

Os níveis 1 e 2 são **puramente organizacionais** — não logam métricas. Só as
iterações (child runs) têm métricas e parâmetros. Se a pessoa A e a pessoa B
rodarem a mesma célula com `test_name = "sem_fe_nf50"`, a parent run
`svd_sem_fe_nf50` é encontrada e **reutilizada** automaticamente — não cria
duplicata.

### O que o usuário define

Apenas dois campos no notebook:

```python
model_type = "svd"          # ← ADAPTAR: "popularity", "svd" ou "mlp"
test_name  = "sem_fe_nf50"  # ← ADAPTAR: descrição do teste (veja regras abaixo)
```

O restante é derivado automaticamente:

| Campo | Derivado de | Exemplo |
| --- | --- | --- |
| `run_group` | `model_type + "_" + test_name` | `"svd_sem_fe_nf50"` |
| `run_name` | `run_group + "_" + timestamp_utc` | `"svd_sem_fe_nf50_20260616T143200Z"` |
| Parent nível 1 | `model_type` | `"svd"` |
| Parent nível 2 | `run_group` | `"svd_sem_fe_nf50"` |

### Regras para `model_type` e `test_name`

**`model_type`** — deve ser um dos valores abaixo (exatamente como escrito):

| Valor | Modelo |
| --- | --- |
| `"popularity"` | Baseline por popularidade |
| `"svd"` | SVD (scikit-surprise) |
| `"mlp"` | MLP com embeddings (PyTorch) |

**`test_name`** — descrição curta do teste. Regras:
- Apenas letras **minúsculas**, números, hífens (`-`) e underscores (`_`)
- Sem espaços, sem maiúsculas, sem caracteres especiais
- Exemplos válidos: `"sem-fe"`, `"com-fe-nf100"`, `"stratificado"`, `"cv-5fold"`
- Exemplos inválidos: `"Sem FE"` ❌, `"com/fe"` ❌, `"nf 50"` ❌

Se o valor for inválido, a função levanta um `ValueError` com mensagem clara:

```
ValueError: test_name 'Sem FE' contém caracteres inválidos.
Use apenas letras minúsculas, números, hífens e underscores (ex: 'sem-fe-nf50').
```

### `phase` — valores e quando usar

| Valor | Quando usar |
| --- | --- |
| `"dev"` | Exploração livre no notebook, sem compromisso de comparação |
| `"baseline"` | Primeira run "oficial" de um modelo, sem tuning |
| `"tuning"` | Iterações de ajuste de hiperparâmetros |
| `"final"` | Run campeã que vai para o Model Registry |

### Exemplo: duas células de teste distintas

```python
# ============================================================
# Célula 0 — configuração (roda uma vez no início do notebook)
# ============================================================
from src.config import get_settings
from src.tracking import configure_mlflow_tracking, log_evaluation_metrics, start_notebook_run

settings = get_settings()

# FIXO — não alterar o nome do experimento
configure_mlflow_tracking(experiment_name="02 - ECOMM_RECSYS - notebook_baselines_training")
```

```python
# ============================================================
# Célula 1 — Teste: SVD sem feature engineering, n_factors=50
# ============================================================
model_type = "svd"           # ← ADAPTAR se mudar de modelo
test_name  = "sem_fe_nf50"   # ← ADAPTAR para descrever este teste

with start_notebook_run(
    model_type, test_name,
    phase="baseline",
    dataset_name="retailrocket",
    params={"n_factors": 50, "random_state": 42},  # ← ADAPTAR — logado automaticamente
) as run:
    # ... treinar modelo ...

    # Métricas — chaves curtas mapeadas automaticamente para os nomes padronizados
    log_evaluation_metrics(
        {"precision": 0.14, "recall": 0.09, "ndcg": 0.17, "hit_rate": 0.28,
         "coverage": 0.38, "revenue": 980.50},
        k=settings.RECOMMENDATION_K,
    )
```

```python
# ============================================================
# Célula 2 — Teste: SVD sem feature engineering, n_factors=100
#            (variação do teste anterior — só muda test_name e params)
# ============================================================
model_type = "svd"            # ← mesmo modelo
test_name  = "sem_fe_nf100"   # ← ADAPTAR: muda aqui para criar novo grupo

with start_notebook_run(
    model_type, test_name,
    phase="baseline",
    dataset_name="retailrocket",
    params={"n_factors": 100, "random_state": 42},  # ← ADAPTAR
) as run:
    # ... treinar modelo ...

    log_evaluation_metrics(
        {"precision": 0.17, "recall": 0.11, "ndcg": 0.20, "hit_rate": 0.32,
         "coverage": 0.41, "revenue": 1120.75},
        k=settings.RECOMMENDATION_K,
    )
```

-----

## 4. Nomenclatura de runs (produção)

Runs oficiais disparadas por `src/training/train.py` não usam `start_notebook_run`.
O padrão de nome permanece:

```
<modelo>_<fase>_<timestamp>
```

Exemplos:

- `popularity_baseline_20260615T143200Z`
- `svd_tuning_20260615T150030Z`
- `mlp_final_20260615T160000Z`

-----

## 5. Modelo, signature e Model Registry

Diretrizes gerais (os detalhes exatos por modelo são definidos quando
`src/models/` for implementado, seguindo o Strategy/Factory):

- **Sempre que possível, logue o modelo com `signature`** via
  `infer_signature`.
    - Para o MLP (PyTorch), a signature é inferida a partir do forward de scoring (`(user_idx, item_idx) -> score`).
    - Para o SVD (`scikit-surprise`), use o wrapper apropriado ou `mlflow.pyfunc`.
    - Para o Popularity, que não tem um "modelo" no sentido clássico (apenas um ranking de itens), logue o artefato de ranking (`mlflow.log_artifact` / `mlflow.log_dict`) — signature não se aplica.
- **Run vs. versão registrada**: toda run que treina um modelo loga o artefato
  (`log_model`/`log_artifact`), mas só a run "campeã" (fase `final`) chama
  `mlflow.register_model(...)`, criando uma nova versão de
  `ecomm-recsys-<model_type>` no Model Registry.
- Promoção para `staging`/`production` é feita via aliases no Model Registry,
  manualmente após avaliação.

-----

## 6. Métricas — convenção de prefixos

Use `log_evaluation_metrics(metrics, k)` (`src/tracking/mlflow_utils.py`) em vez
de `mlflow.log_metrics` diretamente. A função normaliza os nomes, loga
`recommendation_k` como parâmetro e evita erros de digitação.

Chaves aceitas (curtas → nome final no MLflow):

| Chave curta | Nome no MLflow |
| --- | --- |
| `"precision"` | `model.precision_at_k` |
| `"recall"` | `model.recall_at_k` |
| `"ndcg"` | `model.ndcg_at_k` |
| `"hit_rate"` | `model.hit_rate_at_k` |
| `"coverage"` | `business.coverage` |
| `"revenue"` | `business.revenue_at_k` |

Chaves já prefixadas (`"model.*"`, `"business.*"`) e chaves desconhecidas (ex:
`"model.train_loss"`) são passadas sem alteração.

```python
log_evaluation_metrics(
    {
        "precision": 0.18, "recall": 0.12, "ndcg": 0.21, "hit_rate": 0.35,
        "coverage": 0.42,  "revenue": 1530.75,
        "model.train_loss": 0.034,   # MLP: passada sem alteração
        "model.val_loss": 0.041,
    },
    k=settings.RECOMMENDATION_K,
)
```

> Nomes usam `_at_k` (não `@k`) porque `@` não é um caractere válido em nomes
> de métricas/tags do MLflow. O valor de `K` é `Settings.RECOMMENDATION_K` e
> é registrado automaticamente como parâmetro `recommendation_k`.

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
with start_notebook_run(
    model_type, test_name,
    phase="baseline",
    dataset_name="retailrocket",
    params=model.get_params(),  # logado automaticamente na abertura do run
) as run:
    mlflow.set_tag("mlflow.note.content", "SVD sem FE, n_factors=50.")

    # ... treinar modelo ...

    log_evaluation_metrics(model_metrics | business_metrics, k=settings.RECOMMENDATION_K)

    mlflow.log_dict(model_metrics | business_metrics, "metrics.json")
    mlflow.log_dict(id_mappings, "data/id_mappings.json")

    if hasattr(model, "training_history"):
        mlflow.log_artifact("training_curve.png", artifact_path="diagnostics")

    if phase == "final":
        mlflow.register_model(model_uri, f"ecomm-recsys-{model_type}")
        mlflow.log_artifact("docs/model-card.md", artifact_path="model_card")
```

-----

## 8. Checklist por run (notebooks)

- [ ] `configure_mlflow_tracking(experiment_name=...)` chamado no início do notebook
- [ ] `model_type` é um dos valores permitidos: `"popularity"`, `"svd"`, `"mlp"`
- [ ] `test_name` segue `[a-z0-9_-]` — sem espaços, sem maiúsculas
- [ ] Usar `start_notebook_run(model_type, test_name, phase, dataset_name, params=model.get_params())` — nunca `mlflow.start_run` direto nos notebooks
- [ ] `mlflow.note.content` com descrição da run
- [ ] Métricas logadas via `log_evaluation_metrics(metrics, k=settings.RECOMMENDATION_K)` — mínimo 4 para comparar com o MLP
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
from src.tracking import configure_mlflow_tracking, log_evaluation_metrics, start_notebook_run

settings = get_settings()

# FIXO — não alterar
configure_mlflow_tracking(experiment_name="02 - ECOMM_RECSYS - notebook_baselines_training")

# ADAPTAR conforme o teste sendo conduzido
model_type = "popularity"
test_name  = "baseline"

# ADAPTAR: instanciar o modelo para obter os params antes de abrir o run
model = RecommenderFactory.create("popularity", config={})

with start_notebook_run(
    model_type, test_name,
    phase="baseline",
    dataset_name="retailrocket",
    params=model.get_params(),  # FIXO — logado automaticamente na abertura
) as run:
    mlflow.set_tag(
        "mlflow.note.content",
        "Baseline de popularidade — ranking global dos itens mais comprados.",
    )

    # ADAPTAR: treinar o modelo
    model.fit(train_interactions)

    # ADAPTAR: calcular métricas com as funções de avaliação do projeto
    model_metrics = evaluate_model_metrics(model, test_interactions, k=settings.RECOMMENDATION_K)
    business_metrics = evaluate_business_metrics(model, test_interactions, k=settings.RECOMMENDATION_K)

    log_evaluation_metrics(model_metrics | business_metrics, k=settings.RECOMMENDATION_K)  # FIXO
    mlflow.log_dict(model_metrics | business_metrics, "metrics.json")
    mlflow.log_dict(model.get_id_mappings(), "data/id_mappings.json")
    mlflow.log_dict(model.get_ranking(), artifact_file="model/ranking.json")
```

Para o MLP, a diferença principal é logar o `signature` e o artefato do modelo
via `mlflow.pytorch.log_model(model, artifact_path="model", signature=signature)`,
além de `"model.train_loss"`/`"model.val_loss"` por época (passados sem alteração
para `log_evaluation_metrics`) e `diagnostics/training_curve.png`.
