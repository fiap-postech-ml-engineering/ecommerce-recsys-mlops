# 0006 - Diagnóstico: ground truth mais fraco (addtocart) não fecha a lacuna SVD vs Popularity

- **Status:** Aceito
- **Data:** 2026-07-08
- **Contexto relacionado:** `docs/experimentos/0005-diagnostico-instabilidade-score-svd.md`,
  `notebooks/02_experiments.ipynb`, `src/models/svd.py`

## Contexto

[0005](0005-diagnostico-instabilidade-score-svd.md) diagnosticou que o SVD perde do
Popularity em NDCG@10/Hit Rate@10 mesmo depois de corrigir a instabilidade numérica
(`log1p` no score) e testar um grid de 24 combinações até 300 épocas. A causa apontada lá
foi a esparsidade do ground truth de avaliação (`score >= EVENT_WEIGHTS["transaction"]`):
só 533-1.004 usuários elegíveis (test/val) contra um catálogo de ~24 mil itens pós k-core.
0005 deixou como próximo passo não testado: usar um sinal mais fraco (`addtocart`, peso 2)
como ground truth alternativo, pra ter mais exemplos de avaliação e permitir que a
personalização do SVD apareça.

## Decisão

Investigação, não uma mudança de modelo ou de ground truth oficial. Rodado um script
ad-hoc (fora do notebook, mesma metodologia da sessão de 0005) contra o dataset real via
`load_dataset()`, reproduzindo exatamente o pipeline oficial do notebook:
`build_interactions()` → `temporal_split(60/20/20)` → `k_core_filter(min_user_interactions=2,
min_item_interactions=5)` → `apply_log_scaling()` só no caminho do SVD. Treinado
`PopularityRecommender` e a melhor configuração do grid de 0005
(`n_factors=100, n_epochs=300, lr_all=0.02, reg_all=0.02`), e os dois modelos já treinados
foram avaliados sob três thresholds de "relevante" (`score >= peso do evento`, superset
estrito nessa ordem): `transaction` (3, o oficial), `addtocart` (2) e `view` (1), tanto em
`val_df` quanto em `test_df`.

## Resultado medido

| Split | Threshold | Usuários elegíveis | NDCG@10 Popularity | NDCG@10 SVD | Hit Rate@10 Popularity | Hit Rate@10 SVD |
| --- | --- | --- | --- | --- | --- | --- |
| val | transaction | 1.004 | 0,0124 | 0,0003 | 0,0418 | 0,0030 |
| val | addtocart | 1.602 (+59%) | 0,0088 | 0,0001 | 0,0318 | 0,0019 |
| val | view | 3.602 (+259%) | 0,0063 | 0,0001 | 0,0236 | 0,0014 |
| test | transaction | 533 | 0,0089 | 0,0004 | 0,0338 | 0,0038 |
| test | addtocart | 890 (+67%) | 0,0065 | 0,0002 | 0,0247 | 0,0022 |
| test | view | 2.346 (+340%) | 0,0040 | 0,0002 | 0,0171 | 0,0017 |

Afrouxar o ground truth de fato reduz a esparsidade — até +340% mais usuários elegíveis com
`view` em relação a `transaction`. **Mas isso não fecha a lacuna entre Popularity e SVD**:
em todos os 6 cortes testados (2 splits × 3 thresholds), Popularity continua batendo o SVD
por 20x a 90x em NDCG@10, e a razão não melhora conforme o threshold afrouxa — piora no
corte `addtocart` em relação a `transaction`, tanto em `val` quanto em `test`.

## Consequências

- A hipótese de que a esparsidade do ground truth é **causa suficiente** do SVD perder do
  Popularity (levantada em 0005) é refutada: mesmo com muito mais exemplos de avaliação, o
  SVD continua ordenando muito pior que simplesmente recomendar o top-10 global fixo.
- `addtocart` não deve substituir `transaction` como ground truth oficial de report — a
  decisão registrada em `docs/internal/planejamento_inicial.md` continua válida; este teste
  foi só diagnóstico, não uma proposta de mudança de métrica.
- Não há mais hiperparâmetro (épocas já esgotadas em 0005) ou variação de ground truth
  óbvia a testar para o SVD neste dataset — a linha de investigação de tuning do SVD fica
  encerrada por ora.
- Reforça a expectativa, já levantada em 0005, de que o ganho real de personalização deve
  vir do MLP (`03_mlp.ipynb`), que pode explorar features além do colaborativo puro.
