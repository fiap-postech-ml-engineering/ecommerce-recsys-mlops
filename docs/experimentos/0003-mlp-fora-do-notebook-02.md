# 0003 - MLP fora do notebook de baselines

- **Status:** Aceito
- **Data:** 2026-07-03
- **Contexto relacionado:** `notebooks/02_experiments.ipynb`, `notebooks/03_mlp.ipynb`

## Contexto

O PR #18 (`feat/TCF1-152-notebook-experimentos`) implementou o baseline SVD e preencheu
`notebooks/02_experiments.ipynb` com a exploração de Popularity e SVD. O MLP com
embeddings (PyTorch) é o modelo central do desafio, mas tem um ciclo de desenvolvimento
próprio — definição de arquitetura, embeddings de usuário/item/categoria, curva de
treino por época — que não compartilha o formato de "grid de hiperparâmetros +
comparação final" usado para os baselines.

## Decisão

O MLP fica explicitamente fora do escopo de `02_experiments.ipynb`. A exploração do MLP
(arquitetura, hiperparâmetros, curva de treino) vai para um notebook dedicado,
`notebooks/03_mlp.ipynb` — hoje vazio, ainda não implementado.

## Consequências

- `notebooks/02_experiments.ipynb` fica só com os baselines (Popularity, SVD) — mais
  simples de revisar e já mergeável antes do MLP existir.
- A comparação final MLP vs. baselines (exigida pelo desafio, ≥4 métricas) só acontece
  quando `03_mlp.ipynb` for implementado e puder consultar as runs de baseline já
  registradas no experimento `notebook_baselines_training` (ver
  [0002](0002-nomenclatura-experimentos-mlflow.md)).
