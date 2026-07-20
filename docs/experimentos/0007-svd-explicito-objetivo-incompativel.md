# 0007 - SVD explícito otimiza objetivo incompatível com o problema (não é só esparsidade)

- **Status:** Aceito
- **Data:** 2026-07-08
- **Contexto relacionado:** `docs/experimentos/0005-diagnostico-instabilidade-score-svd.md`,
  `docs/experimentos/0006-diagnostico-ground-truth-addtocart.md`, `src/models/svd.py`

## Contexto

[0005](0005-diagnostico-instabilidade-score-svd.md) e [0006](0006-diagnostico-ground-truth-addtocart.md)
diagnosticaram que o `SVDRecommender` (via `scikit-surprise`) perde do `PopularityRecommender`
em todas as métricas de ranking, mesmo depois de corrigir instabilidade numérica, tunar
épocas até 300, e testar um ground truth mais fraco (`addtocart`/`view`). A causa apontada
em 0005/0006 foi a esparsidade do sinal de avaliação. O usuário levantou a hipótese de que
o problema fosse outro — o algoritmo em si — e pediu pesquisa + teste ad-hoc antes de
qualquer mudança de código.

Pesquisa: `scikit-surprise`'s `SVD` otimiza RMSE de **rating explícito** via SGD — a
própria documentação da lib confirma que não há suporte a feedback implícito (nosso
`score` agregado não é um rating de usuário, é uma soma de pesos de evento). Cremonesi,
Koren & Turrin (*"Performance of Recommender Algorithms on Top-N Recommendation Tasks"*,
RecSys 2010) mostram que algoritmos otimizados para minimizar RMSE não necessariamente
performam bem em tarefas de top-N, e que um baseline não-personalizado pode chegar perto
(ou superar) algoritmos sofisticados quando a métrica de treino não é a métrica de
avaliação. Isso sugere que o SVD explícito está otimizando a métrica errada pro problema —
não (só) sofrendo de poucos dados.

## Decisão (investigação, script ad-hoc fora do notebook, mesma metodologia de 0005/0006)

Reproduzido o pipeline oficial (`build_interactions` → `temporal_split(60/20/20)` →
`k_core_filter(min_user=2, min_item=5)` → `apply_log_scaling` pro MF), mantendo `train_df`/
`val_df`/`test_df` idênticos aos usados em 0005/0006. Treinado `PopularityRecommender`, o
melhor `SVDRecommender` do grid de 0005 (`n_factors=100, n_epochs=300, lr_all=0.02,
reg_all=0.02`), e um modelo de fatoração de matriz **implícita** — ALS com ponderação de
confiança (Hu, Koren & Volinsky, *"Collaborative Filtering for Implicit Feedback
Datasets"*, 2008), via a lib `implicit` (instalada só na venv desta sessão, não persistida
em `pyproject.toml` — ver próximos passos). Avaliados os três no mesmo `val_df`/`test_df`,
ground truth oficial (`transaction`).

## Resultado medido

| Split | Modelo | NDCG@10 | HitRate@10 |
| --- | --- | --- | --- |
| val | Popularity | 0,0124 | 0,0418 |
| val | SVD explícito (surprise, melhor grid 0005) | 0,0003 | 0,0030 |
| val | ALS implícito (n_factors=100, alpha=40, iterations=30, sem tuning) | 0,1267 | 0,2490 |
| test | Popularity | 0,0089 | 0,0338 |
| test | SVD explícito | 0,0004 | 0,0038 |
| test | ALS implícito | 0,1059 | 0,2026 |

Trocando **só o algoritmo** (mesmo dataset, split, k-core, catálogo), o resultado vira de
"perde do Popularity por 20-90x" para "ganha por ~10-12x" — sem nenhum tuning de
hiperparâmetro do ALS (valores de literatura, não otimizados). Ver
[0008](0008-als-bpr-itemknn-comparacao.md) para um grid do ALS e comparação com mais duas
famílias de algoritmo implícito.

## Consequências

- A causa raiz principal da perda do SVD não é (só) a esparsidade do ground truth
  (0005/0006) — é o **descasamento entre o objetivo de treino (RMSE de rating explícito) e
  a tarefa real (ranking de top-N sobre feedback implícito)**. A esparsidade é real e
  segue relevante (ver [0009](0009-interpretacao-ndcg-protocolo-avaliacao.md)), mas não é
  suficiente pra explicar um gap de 20-90x que desaparece só trocando o algoritmo.
  `scikit-surprise` continua sendo uma ferramenta inadequada pra este tipo de dataset.
- `SVDRecommender`/`"svd"` (código atual) fica preservado como registro histórico do que
  foi tentado e por que não funcionou — nenhum código muda nesta etapa.
- Fatoração de matriz **implícita** (ALS, e as duas alternativas testadas em 0008) vira o
  candidato real a baseline de matrix factorization do projeto, substituindo ou
  complementando o SVD explícito — decisão de arquitetura ainda em aberto, tratada como
  próximo passo de código (fora do escopo desta etapa de documentação).
