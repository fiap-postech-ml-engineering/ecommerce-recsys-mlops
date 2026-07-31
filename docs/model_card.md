# 🧾 Ficha técnica

- **Nome** — E-commerce RecSys
- **Modelo em Production** — ItemKNN (`n_neighbors=50`, ver `params.yaml`). O modelo central exigido pelo desafio (MLP em PyTorch) foi treinado e comparado, mas não promovido — ver seção "Por que o modelo central (MLP) não está em Production" abaixo.
- **Versão do modelo** — `0.1.0` (`MODEL_VERSION` em `.env.example`)
- **Tipo de modelo** — Collaborative filtering baseado em vizinhança de itens (ItemKNN via `implicit`)
- **Registry** — MLflow Model Registry, `workspace.default.ecomm-recsys-itemknn`, alias `production` (versão 3 no workspace Databricks do time, confirmado em 2026-07-30: `params={model: itemknn, n_neighbors: 50}`, `tags.source=dvc_pipeline` — mesmo pipeline e mesmos hiperparâmetros deste repositório, não um treino manual em paralelo)
- **Baselines comparados** — Popularity, SVD, ALS, BPR, ItemKNN, MLP (embedding-based, PyTorch) — ver `src/models/`

- **Para quais casos o modelo foi projetado**
    - Recomendação de top-N produtos por usuário com base no histórico de interações (clique, carrinho, compra)
    - Suporte a personalização de vitrine/e-mail marketing em e-commerce

## 🎯 Saída

- Lista ordenada de `item_id` (top-`K`, padrão `RECOMMENDATION_K=10`) por `user_id` — ver `RecommendResponse` em `src/api/schemas.py`

## 🔁 Sem acesso ao Databricks? O resultado é o mesmo

Quem roda o pipeline localmente (`MLFLOW_TRACKING_URI=local`, ver README) treina o **mesmo modelo** que está em Production no workspace Databricks do time — mesmo algoritmo (ItemKNN), mesmos hiperparâmetros (`n_neighbors=50` de `params.yaml`), mesmo dataset público (RetailRocket). Não fica menos representativo por ter treinado localmente: o binário exato não é reproduzido (dados de treino têm alguma aleatoriedade de shuffling), mas o comportamento e as métricas obtidas são equivalentes às do modelo real em produção.

# 📊 Métricas

Avaliação em duas fases (ver `docs/internal/planejamento_inicial.md`): seleção de configuração via NDCG@K no `val_df`, avaliação final com as 6 métricas no `test_df` (tocado uma única vez).

## 🏆 Métricas do modelo em Production (`metrics.json`)

| **Métrica** | **Valor** |
|-------------|-----------|
| **`NDCG@K`** (principal) | 0.135 |
| **`Hit Rate@K`** (desempate) | 0.255 |
| **`Precision@K`** | 0.029 |
| **`Recall@K`** | 0.219 |
| **`Coverage`** | 0.180 |
| **`Revenue@K`** | R$ 22.481.796,00 |

## 📈 Comparativo final (`test_df`, `notebooks/03_mlp.ipynb`, seção 7-8)

| Modelo | NDCG@10 | Recall@10 | Precision@10 | HitRate@10 | Revenue@10 | Coverage |
| --- | --- | --- | --- | --- | --- | --- |
| **ItemKNN** | **0,1305** | **0,2122** | **0,0274** | **0,2495** | **22.080.996** | **0,0236** |
| Popularity | 0,0089 | 0,0183 | 0,0039 | 0,0338 | 1.606.560 | 0,0001 |
| MLP | 0,0051 | 0,0108 | 0,0024 | 0,0206 | 1.542.240 | 0,0002 |
| SVD | 0,0015 | 0,0038 | 0,0009 | 0,0075 | 278.616 | 0,0006 |

## ⚖️ Critério de seleção do modelo

Hierarquia de métricas (ver `planejamento_inicial.md`): NDCG@K como critério principal, Hit Rate@K como desempate, Precision@K/Recall@K como monitoramento, Coverage/Revenue@K para justificar valor de negócio (não entram na seleção).

**O ItemKNN foi promovido a Production porque venceu por larga margem em todas as métricas** — não é uma decisão arbitrária, é o resultado direto do critério de seleção acima aplicado ao `test_df`. Detalhes na seção seguinte.

# 🗂️ Dataset utilizado

Dataset **RetailRocket** (eventos de navegação e transações em e-commerce) — ver `docs/internal/planejamento_inicial.md` para o racional de escolha.

- **Input**: `user_id`, `item_id`, `rating/score` (ponderado por tipo de interação: view/addtocart/transaction), `value` (valor monetário)
- **Split**: temporal 60/20/20 (treino/validação/teste) por ordem cronológica — nunca split aleatório, para não vazar interações futuras
- **Filtragem**: k-core filtering para reduzir esparsidade (ver `src/data/filtering.py` e `docs/experimentos/0001-k-core-filtering-esparsidade.md`)

**Volumetria pós k-core filtering** (`docs/experimentos/0001-k-core-filtering-esparsidade.md`): o dataset bruto tem esparsidade extrema (mediana de 1 interação/usuário, 95,8% dos usuários com ≤5 interações). O `k_core_filter()` (`MIN_USER_INTERACTIONS=2`, `MIN_ITEM_INTERACTIONS=5`) reduz o treino de 1.151.589 para 399.022 linhas — 110.199 usuários e 24.016 itens, de um catálogo original de 183.681. O split temporal completo tem 253.353 usuários cold-start no teste (sem nenhuma interação no treino).

# 🤔 Por que o modelo central (MLP) não está em Production

O desafio (`docs/internal/tech_challenge.md`) pede uma rede neural (MLP/embedding-based) em PyTorch como modelo central, comparada a baselines usando ≥ 4 métricas. **Esse requisito foi cumprido integralmente**: o MLP foi implementado (`src/models/mlp.py`), treinado com early stopping (`notebooks/03_mlp.ipynb`, seções 5-7) e comparado ao ItemKNN, Popularity e SVD nas 6 métricas do projeto sobre o mesmo `test_df`.

**O resultado medido, porém, é que o MLP perde para o ItemKNN por ~25x em NDCG@10/Recall@10/Precision@10, ~12x em HitRate@10 e ~14x em Revenue@10 — e nem supera de forma consistente o baseline trivial de Popularity** (só empata em Revenue; perde em todo o resto). Aplicando o critério de seleção definido no projeto (NDCG@K como métrica principal), o ItemKNN foi o modelo promovido a Production — manter o MLP em produção apesar do resultado seria otimizar para cumprir a letra do requisito à custa da qualidade real das recomendações entregues ao usuário.

Esse resultado não é um bug de implementação isolado. É consistente com um achado bem documentado na literatura de RecSys: Dacrema, Cremonesi & Jannach (*"Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches"*, RecSys 2019) mostraram que a maioria dos métodos neurais recentes avaliados no paper eram superados por baselines simples (KNN, popularidade) bem ajustados — o ganho de modelos neurais em RecSys depende fortemente de volume de dados, tuning extensivo e do domínio, e não se materializa automaticamente. Ver `notebooks/03_mlp.ipynb`, seção 8, para a análise completa e a tabela comparativa.

Insumos adicionais documentados em:
- `docs/experimentos/0007-svd-explicito-objetivo-incompativel.md`
- `docs/experimentos/0008-als-bpr-itemknn-comparacao.md`
- `docs/experimentos/0010-mlp-arquitetura-implicita-early-stopping.md`

## 🏗️ Arquitetura utilizada (MLP)

Embeddings de usuário/item + torre MLP, treinado com rótulo binário implícito (presença de interação), negative sampling e early stopping interno (convergiu em 7 de 50 épocas na melhor configuração testada, `hidden_dims=[256, 128, 64]`, `negative_samples=4`). Hiperparâmetros configuráveis via `MLP_EMBEDDING_DIM`/`MLP_HIDDEN_DIMS`/`MLP_EPOCHS`/`MLP_LEARNING_RATE`/`MLP_BATCH_SIZE` em `src/config.py`.

# ⚠️ Limitações

- O modelo depende de histórico de interações prévio do usuário — não resolve cold start (usuários sem interações no treino).
- Avaliado offline (proxy: acerto do item comprado no período de teste); sem teste A/B em produção não há prova de causalidade entre recomendação e conversão.
- Sensível a `n_neighbors` (50 em Production) — não houve busca sistemática desse hiperparâmetro, só o valor default de `Settings.ITEMKNN_N_NEIGHBORS`; vale testar outros valores antes de assumir que é o ótimo.
- Cauda longa do catálogo pouco coberta — Coverage de 18% (ver métricas acima) indica que a maior parte das recomendações se concentra em um subconjunto do catálogo.

## 🧭 Vieses e cuidados

- Viés de popularidade — modelos de collaborative filtering (ItemKNN incluído) tendem a sobrerrepresentar itens já populares nas recomendações, reforçando as escolhas mais comuns em vez de diversificar. O Coverage de 18% do catálogo é consistente com isso, mas o quanto é "concentração saudável em itens relevantes" vs. "viés problemático" não foi quantificado — precisaria de uma métrica de diversidade/novidade dedicada (ex: distribuição de popularidade dos itens recomendados vs. distribuição do catálogo).

## 🚨 Cenários de falha

- **Usuário sem modelo carregado** — a API retorna `503` em `/recommend` quando não há modelo em Production no MLflow Registry (ver `src/api/routes/recommend.py`).
- **Cold start** — usuário sem histórico no treino não tem linha na matriz esparsa; `ItemKNNRecommender.recommend()` retorna lista vazia, sem fallback de popularidade (`src/models/itemknn.py`). A API responde `200` com `recommendations: []`, não um erro — comportamento correto tecnicamente, mas silencioso: o cliente da API não recebe sinal explícito de que caiu em cold start. Fica como melhoria futura definir um fallback (ex: popularidade) para esse caso.
