# 0009 - Interpretação do NDCG@10 medido: protocolo de avaliação e posição do hit

- **Status:** Aceito
- **Data:** 2026-07-08
- **Contexto relacionado:** `docs/experimentos/0006-diagnostico-ground-truth-addtocart.md`,
  `docs/experimentos/0008-als-bpr-itemknn-comparacao.md`, `src/metrics/ranking.py`

## Contexto

Depois de medir NDCG@10 ≈ 0,13-0,16 pros melhores modelos implícitos (0008), veio a
dúvida: esse valor é baixo pro tipo de dataset/projeto? Uma hipótese levantada foi que os
acertos (hits) estariam concentrados nas últimas posições do top-10 — ou seja, o modelo
até recomenda os itens certos, mas rankeados no fim da lista, o que penalizaria o NDCG sem
necessariamente indicar um modelo ruim. Investigado via pesquisa bibliográfica + teste
empírico direto nos dados do projeto (RetailRocket, mesmo pipeline oficial).

## Decisão (pesquisa + teste empírico)

**Pesquisa sobre protocolo de avaliação:** muitos papers reportam NDCG@10/12 de 0,5-0,7
em recomendação de ecommerce, mas frequentemente usando **avaliação por amostragem
negativa** (ex.: ranquear 1 item relevante + 99 itens aleatórios, protocolo popularizado
por papers como NCF/BPR clássicos), não o catálogo inteiro. Krichene & Rendle (*"On
Sampled Metrics for Item Recommendation"*, KDD 2020) mostram que essas métricas amostradas
são **inconsistentes** com a versão exata — não preservam nem a ordem relativa entre
modelos (`A melhor que B` pode inverter dependendo da amostra). O `evaluate_model` deste
projeto sempre ranqueia o **catálogo inteiro** pós k-core (~24 mil itens) — protocolo bem
mais difícil e mais realista, não comparável a números de avaliação amostrada.

**Benchmark real no mesmo dataset, catálogo completo:** a implementação oficial do
GRU4Rec ([hidasib/GRU4Rec_PyTorch_Official](https://github.com/hidasib/GRU4Rec_PyTorch_Official),
rede neural sequencial, referência forte em recsys) reporta, no RetailRocket, com ranking de
catálogo completo: `Recall@10 = 0,42`, `MRR@10 = 0,21`. Só que a tarefa é diferente e mais
fácil — prever o **próximo clique dentro de uma sessão ativa** (sinal sequencial forte,
avaliado por evento), não "o usuário vai comprar X num horizonte de tempo futuro esparso"
como neste projeto (split temporal 60/20/20, só ~1-2 compras por usuário elegível no
período de teste — ver 0006). Números absolutos mais baixos aqui são esperados dado que a
tarefa é mais difícil, não indicam necessariamente um problema de implementação.

**Teste direto da hipótese "hits concentrados no fim da lista":** calculado, pro melhor
modelo (BM25 ItemKNN, ver 0008), em que posição do top-10 cai o primeiro item relevante
encontrado, por usuário elegível:

| | val | test |
| --- | --- | --- |
| Usuários elegíveis | 1.004 | 533 |
| Com ≥1 hit no top-10 | 30,8% | 25,5% |
| Rank mediano do 1º hit (quando ocorre) | 3 | 3 |
| % dos hits em posições 1-3 | 59,2% | 50,7% |
| % dos hits em posições 8-10 | 11,3% | 12,5% |

## Resultado

A hipótese é **refutada**: quando o modelo acerta, o acerto cai perto do topo da lista
(mediana = posição 3, ~50-60% dos hits nas 3 primeiras posições), não no fim (só ~11-13%
nas posições 8-10). O NDCG@10 "parece baixo" porque **~70-75% dos usuários elegíveis não
recebem nenhum hit** no top-10 — um problema de cobertura/recall sob esparsidade extrema
do sinal de compra (catálogo de ~24 mil itens, poucas compras por usuário no período de
teste), não um problema de qualidade de ranqueamento dos itens que são recomendados.

## Consequências

- `K=10` (`Settings.RECOMMENDATION_K`) é confirmado como corte adequado — é o padrão mais
  comum na literatura de recsys (HR@10/NDCG@10 aparece na maioria dos benchmarks citados
  aqui), não uma escolha arbitrária do projeto.
- Aumentar `K` provavelmente melhoraria HitRate/Recall (mais chances de incluir os poucos
  itens relevantes numa lista maior), mas isso não é um "conserto de qualidade de
  ranqueamento" — o gargalo real é cobertura num regime de esparsidade extrema, reforçando
  0006 sob um ângulo novo (posição do hit, não só contagem de usuários elegíveis).
- Os valores de NDCG@10/HitRate@10 medidos em 0007/0008 devem ser lidos como razoáveis
  para o protocolo de avaliação (ranking de catálogo completo) e a tarefa (compra futura
  esparsa) deste projeto — não devem ser comparados diretamente com números de papers que
  usam avaliação amostrada ou tarefas de next-click em sessão.
