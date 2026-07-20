# 0001 - K-core filtering para mitigar esparsidade extrema

- **Status:** Aceito
- **Data:** 2026-07-06
- **Contexto relacionado:** `src/data/filtering.py`, `notebooks/02_experiments.ipynb`

## Contexto

`notebooks/01_eda.ipynb` quantificou a esparsidade extrema do dataset RetailRocket:
mediana de 1 interação por usuário, 95,8% dos usuários com ≤5 interações e 65,8% dos
itens com ≤5 interações (ver notebook para a análise completa). O pipeline de dados
(`load_dataset()` → `build_interactions()` → `temporal_split()`) não tinha nenhum filtro
de densidade mínima — toda a cauda longa entrava crua no treino, e as métricas de
ranking (Precision@10, Recall@10, NDCG@10, Hit Rate@10) ficavam próximas de zero tanto
para o baseline Popularity quanto para o SVD.

## Decisão

Adicionar `k_core_filter()` (`src/data/filtering.py`) como etapa do pipeline, aplicada
**depois** de `temporal_split()`:

- O limiar mínimo de interações é calculado **só a partir do `train_df`** — nunca olhando
  `val_df`/`test_df` — para não vazar dados do futuro no split temporal (mesma regra já
  aplicada ao próprio split).
- Remover um usuário/item pode derrubar outro abaixo do limiar, então o filtro é
  reaplicado até convergir (nenhuma linha a mais é removida).
- `val_df`/`test_df` são restringidos ao mesmo universo de `user_id`/`item_id` que
  sobreviveu no k-core do treino.
- Limiares configurados em `src/config.py` (`Settings`): `MIN_USER_INTERACTIONS=2`,
  `MIN_ITEM_INTERACTIONS=5`. Um `k=5` do lado do usuário removeria ~96% da base (mediana
  já é 1 interação), por isso um corte mais brando (`k=2`) foi escolhido para esse lado;
  do lado do item, `k=5` remove a cauda mais extrema mantendo um catálogo ativo razoável.

## Consequências

- Treino cai de 1.151.589 para 399.022 linhas (110.199 usuários, 24.016 itens de um
  catálogo original de 183.681).
- Métricas de ranking melhoram relativamente para os dois modelos avaliados no `test_df`
  (ex.: Hit Rate@10 do Popularity sobe de 0,0161 para 0,0338; NDCG@10 de 0,0055 para
  0,0089), mas seguem baixas em termos absolutos — o catálogo ainda é grande relativo ao
  volume de interações por usuário.
- O SVD segue perdendo do Popularity em todas as métricas mesmo após o filtro — ver
  [0005](0005-diagnostico-instabilidade-score-svd.md).
