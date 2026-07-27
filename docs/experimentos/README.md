# Decisões de experimentação

Esta pasta registra decisões técnicas tomadas durante a fase de experimentos do projeto
(`notebooks/02_experiments.ipynb` e módulos relacionados em `src/`) — o que foi decidido,
por que, e quais as consequências práticas.

Não duplica o conteúdo de `docs/internal/planejamento_inicial.md` ou
`docs/internal/mlflow/02_mlflow_boas_praticas.md`: quando uma decisão tem origem em algum
desses documentos, o registro aqui referencia o trecho em vez de reescrevê-lo.

## Template

Cada arquivo numerado segue esta estrutura:

```markdown
# NNNN - Título da decisão

- **Status:** Aceito | Proposto
- **Data:** AAAA-MM-DD
- **Contexto relacionado:** notebooks/02_experiments.ipynb (ou módulo/arquivo relevante)

## Contexto

O que motivou a decisão — o problema observado, dados/evidência que embasam.

## Decisão

O que foi decidido, de forma direta.

## Consequências

Efeitos práticos (positivos e negativos) de ter tomado essa decisão.
```

`Status: Proposto` indica uma decisão levantada/diagnosticada mas ainda não implementada —
o registro é atualizado para `Aceito` (ou descartado) quando a correção for de fato aplicada.

## Índice

| # | Decisão | Status |
| --- | --- | --- |
| [0001](0001-k-core-filtering-esparsidade.md) | K-core filtering para mitigar esparsidade extrema | Aceito |
| [0002](0002-nomenclatura-experimentos-mlflow.md) | Nomenclatura e isolamento dos experimentos MLflow | Aceito |
| [0003](0003-mlp-fora-do-notebook-02.md) | MLP fora do notebook de baselines | Aceito |
| [0004](0004-criterio-selecao-hiperparametros-svd.md) | Critério de seleção de hiperparâmetros do SVD | Aceito |
| [0005](0005-diagnostico-instabilidade-score-svd.md) | Diagnóstico: instabilidade do score sem cap no SVD | Aceito parcialmente |
| [0006](0006-diagnostico-ground-truth-addtocart.md) | Diagnóstico: ground truth mais fraco (addtocart) não fecha a lacuna SVD vs Popularity | Aceito |
| [0007](0007-svd-explicito-objetivo-incompativel.md) | SVD explícito otimiza objetivo incompatível com o problema (não é só esparsidade) | Aceito |
| [0008](0008-als-bpr-itemknn-comparacao.md) | ALS, BPR e BM25 ItemKNN: comparação de 3 famílias de CF implícito | Aceito |
| [0009](0009-interpretacao-ndcg-protocolo-avaliacao.md) | Interpretação do NDCG@10 medido: protocolo de avaliação e posição do hit | Aceito |
| [0010](0010-mlp-arquitetura-implicita-early-stopping.md) | Arquitetura do MLP: torre única sobre embeddings concatenados, treino implícito e early stopping interno | Aceito |
