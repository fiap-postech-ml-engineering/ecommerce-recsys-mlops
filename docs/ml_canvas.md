# ML Canvas — Ecommerce Recsys MLOps

> Resumo do "porquê" do projeto. Fonte de verdade completa em
> `docs/internal/planejamento_inicial.md` e `docs/internal/tech_challenge.md`.

## Problema de negócio
Recomendar produtos a partir do comportamento de navegação (dataset RetailRocket).

## Stakeholders
- Time de e-commerce (objetivo: aumento de ticket médio)
- Avaliadores do Tech Challenge

## Restrições
- Split temporal obrigatório (60/20/20), por ordem cronológica — nunca aleatório,
  para não vazar interações futuras para o treino
- ≥ 4 métricas comparando o MLP com os baselines
- Reprodutibilidade via `RANDOM_SEED=42`

## Métrica de negócio principal
- **Revenue@K** — soma do `value` dos itens recomendados que foram comprados
  (`src/metrics/business.py::revenue_at_k`)
- **Coverage** — % do catálogo recomendado (`src/metrics/business.py::coverage`)

## Métricas técnicas
Precision@K, Recall@K, NDCG@K, Hit Rate@K — implementadas em
`src/metrics/ranking.py` (`precision_at_k`, `recall_at_k`, `ndcg_at_k`, `hit_rate_at_k`).

## Fontes de dados
Dataset RetailRocket (`events.csv` + item properties).
Ground truth = interação do tipo `"transaction"`.

## SLOs
Não há requisito de latência formal nesta fase (servir via API é opcional).
Critério de sucesso: o MLP superar os baselines em ≥ 4 métricas.

## Modelos comparados
| Modelo | Descrição | Biblioteca |
| --- | --- | --- |
| PopularityRecommender | Baseline — contagem dos itens mais comprados | Lógica própria |
| SVD | Fatoração de matriz | scikit-surprise |
| MLP embedding-based | Embeddings de usuário/item + rede neural | PyTorch |

## Hiperparâmetros predefinidos
| Parâmetro | Valor |
| --- | --- |
| RANDOM_SEED | 42 |
| TEST_SIZE | 0.2 |
| VALIDATION_SIZE | 0.2 |
| RECOMMENDATION_K | 10 |
| SVD_N_FACTORS | 50 |
| MLP_EMBEDDING_DIM | 64 |
| MLP_HIDDEN_DIMS | [128, 64] |
| MLP_EPOCHS | 50 |
| MLP_LEARNING_RATE | 0.001 |
| MLP_BATCH_SIZE | 256 |