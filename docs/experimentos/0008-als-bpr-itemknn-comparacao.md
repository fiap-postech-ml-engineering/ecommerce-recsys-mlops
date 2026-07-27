# 0008 - ALS, BPR e BM25 ItemKNN: comparação de 3 famílias de CF implícito

- **Status:** Aceito (achado medido); escolha do baseline "oficial" ainda em aberto
- **Data:** 2026-07-08
- **Contexto relacionado:** `docs/experimentos/0007-svd-explicito-objetivo-incompativel.md`,
  `docs/experimentos/0004-criterio-selecao-hiperparametros-svd.md`

## Contexto

[0007](0007-svd-explicito-objetivo-incompativel.md) mostrou que trocar o SVD explícito por
ALS implícito (sem tuning) já vira uma perda de 20-90x numa vitória de ~10-12x sobre o
Popularity. Antes de decidir qual algoritmo formalizar como baseline de matrix
factorization do projeto, era preciso saber (a) se esse resultado do ALS é sensível à
escolha de hiperparâmetro, e (b) se existe alternativa melhor — a lib `implicit` (já
instalada pro teste de 0007) também traz BPR (ranking pairwise) e métodos de vizinhança
(ItemKNN), sem exigir dependência nova.

## Decisão (investigação, script ad-hoc, mesma metodologia de 0007)

Rodado um grid pequeno de ALS (`n_factors∈{50,100}, regularization∈{0.01,0.1},
alpha∈{15,40}`, 8 combinações, `iterations=20`), selecionado por NDCG@10 no `val_df`
(desempate por HitRate@10) — mesmo critério de seleção documentado em
[0004](0004-criterio-selecao-hiperparametros-svd.md). Testados também, com hiperparâmetros
de literatura sem tuning: `BayesianPersonalizedRanking` (BPR — loss de ranking pairwise,
`factors=100, iterations=100, learning_rate=0.01, regularization=0.01`) e
`BM25Recommender` (ItemKNN — vizinhança item-item ponderada por BM25, sem fatores latentes,
`K=50` vizinhos — parâmetro da lib, sem relação com `Settings.RECOMMENDATION_K=10`, que
continua sendo o corte de avaliação usado em todos os modelos). Todos avaliados no mesmo
`val_df`/`test_df`, ground truth `transaction`, `K=10` (`RECOMMENDATION_K`).

## Resultado medido

| Split | Modelo | NDCG@10 | HitRate@10 | Precision@10 | Recall@10 |
| --- | --- | --- | --- | --- | --- |
| val | Popularity | 0,0124 | 0,0418 | 0,0049 | 0,0204 |
| val | SVD explícito | 0,0003 | 0,0030 | 0,0003 | 0,0003 |
| val | ALS (melhor do grid: n_factors=100, regularization=0.1, alpha=40) | 0,1308 | 0,2530 | 0,0318 | 0,2047 |
| val | BPR (default, sem tuning) | 0,0645 | 0,1205 | 0,0169 | 0,0910 |
| val | **BM25 ItemKNN (K=50)** | **0,1616** | **0,3078** | **0,0372** | **0,2538** |
| test | Popularity | 0,0089 | 0,0338 | 0,0039 | 0,0183 |
| test | SVD explícito | 0,0004 | 0,0038 | 0,0004 | 0,0000 |
| test | ALS (melhor do grid) | 0,1084 | 0,2026 | 0,0229 | 0,1631 |
| test | BPR (default) | 0,0522 | 0,1163 | 0,0126 | 0,0773 |
| test | **BM25 ItemKNN** | **0,1349** | **0,2552** | **0,0289** | **0,2191** |

O grid do ALS moveu o NDCG@10 (val) de 0,1267 (config inicial do 0007) para 0,1308 (melhor
de 8 combinações) — melhora real, mas pequena (~3%) perto do ganho de trocar de algoritmo
(SVD→qualquer implícito, ~300-500x). **BM25 ItemKNN, mesmo sem nenhum tuning, supera o ALS
tunado em todas as métricas**, tanto em `val` quanto em `test`.

## Consequências

- Confirma que o resultado de 0007 não foi sorte de hiperparâmetro — 3 famílias de
  algoritmo de CF implícito diferentes (fatoração com confiança, ranking pairwise,
  vizinhança) superam o Popularity e o SVD explícito por uma margem grande.
- Achado consistente com a literatura (Cremonesi et al. 2010; Dacrema et al., RecSys 2019,
  *"Are we really making much progress?"*): métodos de vizinhança simples e bem ajustados
  frequentemente superam fatoração de matriz mais sofisticada — vale considerar ItemKNN
  como candidato sério, não só uma curiosidade.
- **Nenhum dos 3 modelos teve tuning exaustivo** — só o ALS teve grid (pequeno); BPR usou
  um único config de literatura; o `K` de vizinhança do BM25 ItemKNN não foi variado. A
  ordem atual (ItemKNN > ALS > BPR) pode mudar com mais tuning, especialmente do BPR.
- A escolha de qual(is) modelo(s) formalizar como baseline "oficial" de matrix
  factorization/CF do projeto (substituir o SVD, adicionar ao lado, manter mais de um) é
  uma decisão de arquitetura ainda em aberto — tratada como próximo passo de código, fora
  do escopo desta etapa de documentação.
