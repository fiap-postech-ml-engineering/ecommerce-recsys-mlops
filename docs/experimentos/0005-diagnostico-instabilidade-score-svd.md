# 0005 - Diagnóstico: instabilidade do score sem cap no SVD

- **Status:** Aceito parcialmente (correção implementada e medida; resolveu a
  instabilidade numérica diagnosticada, mas não fez o SVD superar o Popularity —
  ver seção "Resultado medido")
- **Data:** 2026-07-06
- **Contexto relacionado:** `src/models/svd.py`, `src/data/preprocessor.py`
  (`EVENT_WEIGHTS`, `apply_log_scaling`), `notebooks/02_experiments.ipynb`

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

## Decisão (implementada)

1. `apply_log_scaling()` (`src/data/preprocessor.py`) aplica `log1p` ao `score` só no
   caminho de treino do SVD — função utilitária isolada, não uma nova Strategy (o
   Strategy do projeto varia modelo, não pré-processamento) e não dentro do modelo (que
   deve receber dados já prontos). `PopularityRecommender` e o schema Pandera continuam
   com o `score` bruto.
2. `SVDRecommender` passou a expor `lr_all`/`reg_all` como parâmetros de config (antes só
   `n_factors`/`n_epochs` eram configuráveis), permitindo variá-los no grid.
3. `notebooks/02_experiments.ipynb` (seção 5) treina sobre
   `train_df_svd = apply_log_scaling(train_df)` e testa um grid completo (produto
   cartesiano via `itertools.product`) de `n_factors` (25/50/100) × `n_epochs` (100/300)
   × `lr_all` (0,005/0,02) × `reg_all` (0,02/0,1) — 24 combinações.

## Resultado medido

**A instabilidade numérica foi confirmada e removida.** Comparação direta, mesma
configuração (`n_factors=50, n_epochs=20, lr_all=0,005` default), medida em `val_df`:

| Configuração | NDCG@10 | Hit Rate@10 |
| --- | --- | --- |
| Score bruto (antes da correção) | 0,0000 | 0,0000 |
| Score log1p (depois da correção) | 0,0004 | 0,0030 |
| Melhor do grid novo (`nf=100, ep=300, lr_all=0,02, reg_all=0,02`) | 0,0011 | 0,0080 |

Sem o `log1p`, o SVD **nunca acertava uma única recomendação** em `val_df` (NDCG e Hit
Rate exatamente zero) — não só instável com `lr_all` alto, mas degenerado mesmo no
default. Com o `log1p`, o modelo passa a aprender algo real: a correlação Spearman
`bi` × popularidade real ficou em 0,42–0,45 nas configurações testadas, em linha com a
expectativa de 100-300 épocas já registrada nesta ficha, e todas as 24 combinações
do grid treinaram sem `NaN` (inclusive com `lr_all=0,02`, que divergia antes).

**Mas o SVD ainda perde do Popularity em NDCG@10 e Hit Rate@10** (comparativo final do
notebook, `test_df`): Popularity ≈ 0,0055 / 0,0161 vs. melhor SVD do grid ≈ 0,0001 /
0,0006. A causa não é mais a instabilidade numérica (essa foi eliminada) — é outra,
levantada nesta mesma investigação:

- **O sinal de avaliação é extremamente esparso.** Só 1.004 usuários (de ~110 mil no
  k-core de treino) têm alguma compra em `val_df`, contra um catálogo de ~24 mil itens
  pós k-core. Nesse regime, o Popularity "acerta" 42 desses 1.004 usuários (Hit
  Rate@10 ≈ 4,2%) só por concentrar toda a massa de recomendação nos 10 itens mais
  vendidos globalmente — uma aposta de baixa variância que paga bem quando há tão pouco
  histórico de compra por usuário para qualquer modelo personalizar em cima. O SVD, ao
  espalhar a atenção entre usuários (personalização real), diluiu a chance de acertar
  esses poucos hits concentrados.
- Não há indício de bug de implementação: recomendações inspecionadas manualmente
  mostram o SVD gerando listas diferentes e plausíveis por usuário (ao contrário do
  Popularity, que repete o mesmo top-10 fixo para todos) — o modelo funciona
  mecanicamente, só não converge para os itens certos com o volume de dados disponível.

## Consequências

- As duas causas originalmente diagnosticadas (outlier no score, poucas épocas) eram
  reais e a correção as resolveu como esperado — o `log1p` não é uma correção a
  descartar, é pré-requisito para o SVD aprender qualquer coisa neste dataset.
- Superar o Popularity, porém, provavelmente exige atacar a causa nova identificada
  aqui (esparsidade do sinal de compra em relação ao catálogo), não mais hiperparâmetros
  do SVD. Candidatos a explorar em um próximo diagnóstico: usar sinais mais fracos que
  `transaction` como ground truth (ex.: `addtocart`) para ter mais exemplos de avaliação,
  ou aceitar que num dataset tão esparso um SVD puro não bate um baseline de popularidade
  e que o ganho real de personalização só aparece no MLP (`03_mlp.ipynb`), que pode usar
  features de conteúdo além de colaborativo puro.
- **Atualização ([0006](0006-diagnostico-ground-truth-addtocart.md)):** o candidato acima
  (ground truth mais fraco) foi testado e refutado como causa suficiente — afrouxar para
  `addtocart` reduz a esparsidade, mas não fecha a lacuna entre Popularity e SVD. Reforça
  a alternativa: o ganho de personalização deve vir do MLP.
- **Atualização ([0007](0007-svd-explicito-objetivo-incompativel.md)):** a causa raiz
  principal não era (só) esparsidade — era o algoritmo. `scikit-surprise`'s `SVD` otimiza
  RMSE de rating explícito, não ranking implícito. Trocando só o algoritmo (mesmo
  pipeline/split/dados) por fatoração de matriz implícita (ALS), o resultado vira de
  "perde do Popularity por 20-90x" para "ganha por ~10-12x" — ver 0007/0008 para a
  investigação completa e mais duas famílias de algoritmo testadas.
- `n_factors=100, n_epochs=300, lr_all=0,02, reg_all=0,02` é a melhor configuração do
  grid novo e deve ser a referência para `train.py`/`params.yaml` quando o SVD for
  formalizado — mesmo perdendo do Popularity, é estritamente melhor que qualquer
  configuração testada com o score bruto.
