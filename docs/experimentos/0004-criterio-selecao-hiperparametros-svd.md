# 0004 - Critério de seleção de hiperparâmetros do SVD

- **Status:** Aceito
- **Data:** 2026-07-03
- **Contexto relacionado:** `notebooks/02_experiments.ipynb`, `docs/internal/planejamento_inicial.md`

## Contexto

`notebooks/02_experiments.ipynb` testa 4 variações de `n_factors`/`n_epochs` do
`SVDRecommender` (`nf25_ep20`, `nf50_ep20`, `nf100_ep20`, `nf50_ep50`), todas avaliadas
no mesmo `val_df` (split temporal não permite cross-validation embaralhada — ver notebook,
seção 5). É preciso um critério objetivo para escolher qual variação vira a configuração
"final", avaliada uma única vez no `test_df`.

O critério em si (NDCG@K como principal, Hit Rate@K como desempate) já está justificado
em `docs/internal/planejamento_inicial.md` — NDCG penaliza acertos no fim da lista
(avalia qualidade da ordenação), enquanto Hit Rate@K é mais interpretável para negócio e
usado só quando o NDCG está próximo entre variações. Este registro documenta que o
notebook de experimentos aplica esse critério concretamente ao grid do SVD.

## Decisão

Seleção da melhor configuração do SVD:

```python
best_svd_name = max(
    svd_results,
    key=lambda name: (svd_results[name]["metrics"]["ndcg"], svd_results[name]["metrics"]["hit_rate"]),
)
```

NDCG@10 como chave primária de ordenação, Hit Rate@10 como desempate — a tupla `(ndcg,
hit_rate)` do `max()` implementa isso diretamente.

## Consequências

- Escolha da configuração "final" é reprodutível e sem ambiguidade — qualquer pessoa que
  rodar o mesmo grid chega à mesma configuração vencedora.
- O critério é aplicado sempre no `val_df`; o `test_df` só é tocado uma vez, na avaliação
  final da configuração já escolhida (evita overfitting ao conjunto de teste).
