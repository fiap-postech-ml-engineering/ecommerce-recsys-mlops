# 0005 - Diagnóstico: instabilidade do score sem cap no SVD

- **Status:** Proposto (diagnóstico feito, correção ainda não implementada)
- **Data:** 2026-07-06
- **Contexto relacionado:** `src/models/svd.py`, `src/data/preprocessor.py` (`EVENT_WEIGHTS`)

## Contexto

Mesmo depois do k-core filtering ([0001](0001-k-core-filtering-esparsidade.md)), o SVD
perde do baseline Popularity em **todas** as métricas de ranking, no `val_df` e no
`test_df` — resultado atípico, já que um modelo personalizado normalmente supera um
baseline não-personalizado.

Investigação feita nesta sessão (script ad-hoc contra os dados reais, fora do notebook)
encontrou duas causas concretas, não um bug de implementação em
`SVDRecommender._score_all_items()`/`recommend()` (a fórmula do estimador bate com a do
`scikit-surprise`: `global_mean + bu[u] + bi[i] + qi[i]·pu[u]`):

1. **`score` sem cap alimentando uma regressão sensível a outlier.** `score`
   (`src/data/preprocessor.py`, soma de `EVENT_WEIGHTS` por par usuário-item) não tem
   limite superior. No `train_df` pós k-core: mediana = 1, mas máximo = 308 (provável
   usuário/bot que revisitou o mesmo item centenas de vezes). O `Reader(rating_scale=...)`
   do SVD usa esse range bruto. Subir `lr_all` de 0,005 (default) para 0,02 — buscando
   convergência mais rápida — faz o treino **divergir** (`bi`/`bu` viram `NaN`):
   instabilidade numérica clássica de gradiente explodindo por causa do outlier no alvo
   de regressão.
2. **Poucas épocas no grid atual.** `notebooks/02_experiments.ipynb` testa só 20-50
   épocas. Correlação de Spearman entre o item bias aprendido (`bi`) e a popularidade
   real do item (soma de `score` por item):

   | Épocas | Correlação (`bi` × popularidade real) |
   | --- | --- |
   | 20 (grid atual) | 0,30 |
   | 100 | 0,43 |
   | 300 | 0,45 |

   Ou seja, com o número de épocas testado hoje, o próprio item bias — o termo mais
   simples do modelo — mal reflete a popularidade real dos itens.

## Decisão (proposta, não implementada)

1. Capar ou aplicar `log1p` no `score` antes de treinar o SVD (remove a instabilidade
   numérica do outlier).
2. Só depois disso, aumentar `n_epochs` no grid de `notebooks/02_experiments.ipynb`
   (testar 100-300) e retestar `lr_all` mais alto — sem o cap, um learning rate maior
   quebra o treino, então a ordem importa.

## Consequências (esperadas, ainda não medidas)

- Deve melhorar a correlação `bi` ↔ popularidade real e permitir treinar com learning
  rate mais alto sem divergência, convergindo mais rápido dentro do mesmo orçamento de
  épocas.
- Impacto real nas métricas de ranking (Precision/Recall/NDCG/Hit Rate) ainda não medido
  — a correção não foi implementada nem testada ponta a ponta. Este registro deve ser
  atualizado para `Aceito` (com números) quando isso acontecer, ou revisado/descartado se
  a correção não resolver o problema.
