# 0010 - Arquitetura do MLP: torre única sobre embeddings concatenados, treino implícito e early stopping interno

- **Status:** Aceito
- **Data:** 2026-07-20
- **Contexto relacionado:** `src/models/mlp.py`, `docs/experimentos/0007-svd-explicito-objetivo-incompativel.md`,
  `docs/experimentos/0003-mlp-fora-do-notebook-02.md`

## Contexto

`src/models/mlp.py` era um stub (`fit`/`recommend`/`get_params` levantavam
`NotImplementedError`). Implementá-lo de verdade exige decidir: (1) como combinar os
embeddings de usuário e item, (2) que objetivo de treino usar dado que as interações são
feedback implícito (não uma nota explícita), e (3) como fazer early stopping — critério de
avaliação explícito do desafio — sem que `fit(interactions)` deixe de aceitar exatamente a
mesma assinatura usada por todos os outros modelos (`BaseRecommender.fit`, Strategy).

O ADR 0007 já estabeleceu que tratar as interações como regressão de rating explícito
(o que o SVD faz) otimiza o objetivo errado e perde feio para o Popularity — o MLP precisa
seguir o mesmo paradigma implícito que ALS/BPR/ItemKNN, não repetir o erro do SVD.

## Decisão

- **Arquitetura**: uma única torre MLP sobre embeddings de usuário e item concatenados
  (braço "MLP" do NCF, sem braço GMF em paralelo) — `Settings` só expõe um par
  `MLP_EMBEDDING_DIM`/`MLP_HIDDEN_DIMS`, não uma segunda configuração de fatores latentes;
  adicionar uma segunda torre exigiria inventar hiperparâmetros que o projeto não define.
- **Objetivo de treino**: classificação binária implícita — positivos são interações
  observadas, negativos são itens não vistos amostrados aleatoriamente por usuário
  (`MLP_NEGATIVE_SAMPLES` por positivo, amostrados uma vez no início do `fit()`, não
  re-amostrados a cada época), loss `BCEWithLogitsLoss` sobre o logit cru (sem sigmoid —
  ranking por logit é equivalente a ranking por probabilidade, é monotônico).
- **Reprodutibilidade**: `torch.manual_seed` + `np.random.default_rng`, ambos a partir de
  `random_state` (default `Settings.RANDOM_SEED`) — nenhuma seed solta no código.
- **Early stopping**: `fit(self, interactions)` mantém a assinatura idêntica à de todos os
  outros modelos (nenhum parâmetro extra de validação) — internamente, um holdout aleatório
  90/10 sobre as próprias `interactions` de treino gera a curva `training_history`
  (`train_loss`/`val_loss`) usada como sinal de parada. Esse holdout é **distinto** do
  `val_df` do split temporal do pipeline: `val_df` continua sendo usado só para métricas de
  ranking (precision/recall/ndcg/hit_rate) no notebook, nunca é misturado ao treino do MLP.
  Após `MLP_EARLY_STOPPING_PATIENCE` épocas sem melhora de `val_loss` (delta mínimo
  `1e-4`), o treino para e os pesos da melhor época (não da última) são restaurados.
- **Cold-start**: `user_id` desconhecido do treino retorna lista vazia em `recommend`, sem
  fallback de viés — mesmo comportamento de `ALSRecommender`/`BPRRecommender`/
  `ItemKNNRecommender`, não o fallback do SVD explícito.

## Consequências

- `get_params()` reporta `epochs` (máximo configurado) e `epochs_trained` (real, após
  early stopping) separadamente — permite logar no MLflow tanto o teto configurado quanto
  quantas épocas de fato rodaram antes de convergir.
- `training_history` fica disponível na instância para o notebook plotar a curva de loss
  (`diagnostics/training_curve.png`, ver `docs/internal/mlflow/02_mlflow_boas_praticas.md`
  seção 7).
- Dois novos hiperparâmetros em `Settings`: `MLP_NEGATIVE_SAMPLES` (default 4) e
  `MLP_EARLY_STOPPING_PATIENCE` (default 5) — seguem a mesma convenção de todo modelo
  pós-SVD (hiperparâmetros centralizados em `src/config.py`, não hardcoded).
- Sem braço GMF, o modelo não captura interações puramente bilineares tão eficientemente
  quanto um NeuMF completo capturaria — aceito como trade-off, dado que o desafio pede "MLP
  funcional com early stopping", não uma arquitetura NCF completa.
