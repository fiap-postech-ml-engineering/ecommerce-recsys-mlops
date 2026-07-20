# 0002 - Nomenclatura e isolamento dos experimentos MLflow

- **Status:** Aceito
- **Data:** 2026-07-06
- **Contexto relacionado:** `notebooks/02_experiments.ipynb`, `docs/internal/mlflow/02_mlflow_boas_praticas.md`

## Contexto

`docs/internal/mlflow/02_mlflow_boas_praticas.md` já documenta os três experimentos
canônicos do projeto dentro da pasta compartilhada `/Shared/mlflow_ecomm_recsys`
(`notebook_baselines_training`, `notebook_mlp_training`, `Production` — ver essa doc,
seção 1, para a tabela completa de quando usar cada um).

Apesar disso documentado, a célula de setup de `notebooks/02_experiments.ipynb` chamava
`configure_mlflow_tracking()` **sem** o argumento `experiment_name`. Isso fazia a função
cair no default de `Settings.MLFLOW_EXPERIMENT_NAME`
(`.../02 - ECOMM_RECSYS - Production`) — ou seja, todo run de exploração de baseline
(`phase="baseline"`/`"tuning"`) rodado a partir do notebook estava sendo logado no
experimento de **produção**, junto com as runs oficiais que `src/training/train.py`
disparará no futuro.

## Decisão

A célula de setup do notebook passa `experiment_name` explicitamente:

```python
configure_mlflow_tracking(
    experiment_name="/Shared/mlflow_ecomm_recsys/02 - ECOMM_RECSYS - notebook_baselines_training"
)
```

`src/config.py` (`Settings.MLFLOW_EXPERIMENT_NAME`) **não foi alterado** — continua
apontando para o experimento de produção por padrão, como já era; a correção é só no
notebook, que precisa declarar explicitamente que é exploração.

## Consequências

- Runs de exploração de baseline (Popularity, SVD) deixam de poluir o experimento de
  produção.
- Validado localmente nesta sessão (backend de arquivo do MLflow, já que não há
  credenciais Databricks neste ambiente) — o experimento criado tem exatamente o nome
  `notebook_baselines_training` esperado.
- Ainda não validado contra o Databricks real (produção) — pendente para quem rodar o
  notebook com `.env` configurado (ver `docs/internal/mlflow/01_configuracao_mlflow.md`).
