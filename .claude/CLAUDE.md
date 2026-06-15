# ecommerce-recsys-mlops

Sistema de recomendação de produtos para e-commerce (Tech Challenge FIAP — MLE, Fase 2).
Pipeline ponta a ponta: PyTorch (MLP embedding-based) + Scikit-Learn (baselines) + MLflow
(tracking/registry) + DVC (versionamento de dados e pipeline) + FastAPI (serving) + uv (deps).

As decisões de produto/modelagem já tomadas estão documentadas em:

- `docs/internal/tech_challenge.md` — requisitos do desafio, etapas obrigatórias e critérios
  de avaliação. Consulte antes de propor mudanças de escopo ou estrutura.
- `docs/internal/planejamento_inicial.md` — dataset (RetailRocket), EDA, formato dos dados,
  estratégia de split, hiperparâmetros, métricas e justificativa dos design patterns.

Trate esses dois documentos como fonte de verdade. Não duplique o conteúdo deles aqui —
apenas referencie.

## Comandos essenciais (Makefile)

| Comando | O que faz |
| --- | --- |
| `make requirements` | `uv sync` — instala dependências |
| `make test` | `pytest tests/ -v --no-cov` |
| `make test-cov` | pytest com cobertura (mínimo 70%, relatório HTML) |
| `make lint` / `make lint-fix` | `ruff check` / com `--fix` |
| `make format-fix` | `ruff format` |
| `make check` | lint + format + test (rodar antes de finalizar qualquer tarefa) |
| `make init` | sobe a API com `uvicorn --reload` |

Ainda não existem comandos DVC/MLflow/Docker (`dvc.yaml`, `docker-compose.yml` e `Dockerfile`
estão vazios/ausentes — fazem parte da Etapa 3). Não invente comandos que não existem.

## Arquitetura alvo e estado atual

Ordem de implementação planejada (ver seção "Estrutura de desenvolvimento" do
`planejamento_inicial.md` para o raciocínio completo):

```
EDA (notebooks/01_eda.ipynb)
  → src/data/ (loader, preprocessor, split)
  → src/tracking/ (mlflow_utils)
  → src/models/ (base, factory, popularity, svd, mlp)
  → notebooks/02_experiments.ipynb
  → src/training/train.py
  → DVC pipeline + Docker
```

Hoje, `src/data/`, `src/models/`, `src/evaluation/`, `src/tracking/` e `src/training/` são
stubs vazios. `src/config.py` (Pydantic Settings, `get_settings()`) e `src/api/app.py`
(FastAPI básica com healthcheck) já estão implementados.

### Design patterns obrigatórios

- **Strategy** — `BaseRecommender` (ABC) define a interface comum `fit`, `recommend`,
  `get_params`. Todos os modelos (Popularity, SVD, MLP) implementam essa interface, de modo
  que `train.py` e `evaluate.py` nunca precisem de `if/elif` por tipo de modelo.
- **Factory Method** — `RecommenderFactory.create(name, config)` instancia o modelo correto
  a partir de uma string (vinda de `params.yaml` do DVC), permitindo trocar de modelo sem
  alterar `train.py`.

Ao implementar qualquer modelo novo, siga essas duas interfaces — não crie atalhos
específicos por modelo fora delas.

### Configuração e hiperparâmetros

Todos os hiperparâmetros (RANDOM_SEED=42, TEST_SIZE, VALIDATION_SIZE, RECOMMENDATION_K,
SVD_N_FACTORS, MLP_EMBEDDING_DIM, MLP_HIDDEN_DIMS, MLP_EPOCHS, MLP_LEARNING_RATE,
MLP_BATCH_SIZE etc.) já estão definidos em `src/config.py` via `Settings`/`get_settings()`.
Use sempre essa fonte — não hardcode valores que já existem ali.

## Modelagem de dados e split

- **Input**: `user_id`, `item_id`, `rating/score` (ponderado por tipo de interação),
  `value` (valor monetário, usado nas métricas de negócio).
- **Output**: lista ordenada de top-N `item_id` recomendados por `user_id`.
- **Split temporal 60/20/20** (treino/validação/teste) por ordem cronológica — split
  aleatório vaza interações futuras para o treino e infla métricas artificialmente. Nunca
  fazer `train_test_split` aleatório nesse projeto.

## Métricas

- **Modelo**: Precision@K, Recall@K, NDCG@K, Hit Rate@K.
- **Negócio**: Coverage (% do catálogo), Revenue@K (soma do `value` dos itens recomendados
  e comprados).
- O desafio exige comparar o MLP com baselines usando ≥ 4 dessas métricas.

## Convenções de código

- Python 3.13, type hints obrigatórios em funções públicas, docstrings Google style.
- Funções ≤ 20 linhas (requisito do desafio).
- Ruff já configurado (line-length=99, regras E/F/I/N/W) — rode `make lint` e
  `make format-fix` antes de finalizar qualquer alteração. Pre-commit já ativo.
- Testes em `tests/`, markers `unit`/`integration`/`model`/`api`/`slow`, cobertura mínima
  70% sobre `src/`.
- Reprodutibilidade: usar sempre `RANDOM_SEED` de `Settings`, não fixar seeds soltas no
  código.

## Git: commits, branches e PRs

- Commits no formato `<tipo>(<escopo>): <descrição no imperativo, em português>`
  (`feat`/`fix`/`refactor`/`test`/`docs`/`chore`), pequenos e por responsabilidade lógica.
- Branches: `<tipo>/<slug-descritivo>`, com `-<ID-da-tarefa>` opcional quando houver
  card no Kanban (ex: `feat/TCF1-52-churn-dataset-dataloader`).
- Ao abrir PR, usar a skill `pr-description` para gerar título e corpo em Markdown.
- **Fluxo de trabalho com o Claude Code**: implementar → usuário valida → só rodar
  `make check` e criar commit/PR quando solicitado explicitamente ("commita", "abre
  PR", etc). Se um novo pedido de implementação chegar antes do ciclo anterior ser
  fechado (check + commit), o Claude deve avisar que o escopo mudou e sugerir
  finalizar o ciclo atual antes de seguir.
- **Não incluir** a linha `Co-Authored-By: Claude ...` (ou qualquer menção ao Claude
  Code) no corpo dos commits.

## Pendências conhecidas

- `pyproject.toml` ainda não tem `torch`, `scikit-learn`, `mlflow`, `dvc`,
  `scikit-surprise` — adicionar somente quando a implementação dos modelos exigir.
- `Dockerfile`, `docker-compose.yml`, `dvc.yaml` vazios — parte da Etapa 3.
- `README.md` ainda tem placeholders Lorem Ipsum — atualizar na Etapa 4.

Não trate essas pendências como bugs a corrigir proativamente — fazem parte das próximas
etapas do desafio e devem ser endereçadas na ordem do pipeline descrito acima.
