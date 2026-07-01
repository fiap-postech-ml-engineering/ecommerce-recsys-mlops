Uma empresa de e-commerce precisa de um sistema de recomendação de produtos baseado no comportamento de navegação dos usuários.

# Datasets que podemos utilizar

- https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset *← Claude mandou escolher esse pelo realismo e volumetria*

# Direcionais para o EDA
- Quantidade dos eventos (Distruibuição)
- Quantos usuários tem poucos transactions (1-5) (desconsiderar)
- Distribuição da coluna value (outliers, média, mediana, missing values)
- Distribuição de eventos ao longo do tempo — verificar sazonalidade que possa comprometer o split 60/20/20 (isolar sazonalidades)
- Volume de usuários com interações nos 3 períodos (treino, validação, teste) — usuários que só aparecem no teste são cold start e não conseguimos recomendar


# Modelagem dos dados

Informações mínimas que o dataset deve conter (INPUT):
- **user_id:** identificador único do usuário (númerico sequencial)
- **item_id:** identificador único do produto
- **rating/score:** baseado no comportamento do usuário (definido arbitrariamente)
    - Exemplo: 1 se o usuário clicou no produto, 2 se adicionou ao carrinho, 3 se comprou
    - Podemos utilizar somente a interação com o maior peso (no caso acima a compra), mas entendo que a soma das interações pode ser mais rica para o modelo aprender padrões de comportamento
- **value:** valor monetário do item (vamos usar para as métricas de negócio)

Informações que o modelo vai retornar (OUTPUT):
- Para cada usuário, uma lista ordenada dos top-N produtos recomendados (item_id)

Assim:
| user_id | recommendations |
| ------- | --------------- |
| 12345   | [67890, 11223, 44556, ...] |
| 67891   | [44556, 99001, 33445, ...] |

Ou assim:
| user_id | item_id | rank |
| ------- | ------- | ---- |
| 12345   | 67890   | 1 |
| 12345   | 11223   | 2 |
| 12345   | 44556   | 3 |


# Estratégia de Split

O dataset tem timestamp — o split é feito por **ordem cronológica**, não aleatoriamente. Split aleatório vazaria interações futuras para o treino, inflando artificialmente as métricas.

**Premissa:** se o modelo acerta prever o que o usuário comprou no período de teste, é razoável assumir que uma recomendação daquele item teria influenciado a compra. É uma aproximação — sem teste A/B em produção não há como provar causalidade, mas é o padrão aceito para avaliação offline de RecSys.


```
|←────── treino (60%) ──────→|←── validação (20%) ──→|←── teste (20%) ──→|
      interações antigas         intermediárias          mais recentes
              ↑                        ↑                       ↑
       aprende padrões       ajusta hiperparâmetros     avaliação final
```

- **Treino:** interações mais antigas — o modelo aprende os padrões de comportamento
- **Validação:** interações intermediárias — usado para ajustar hiperparâmetros e early stopping
- **Teste:** interações mais recentes — avaliação final, tocado apenas uma vez


# Modelos que podemos utilizar

|Modelo | Descrição | Biblioteca |
| --- | --- | --- |
|PopularityRecommender | Tem o mesmo papel do "DummyClassifier". Contagem dos itens mais comprados | Lógica própria |
|SVD/NMF | Utiliza técnicas de fatoração de matriz para recomendação | surprise |
|MLP embedding-based | Utiliza o pytorch para criar embeddings e definir uma arquitetura neural | PyTorch |

# Hiperparâmetros predefinidos no config.py

| Parâmetro | Valor | Descrição |
| --- | --- | --- |
| RANDOM_SEED | 42 | Semente para reprodutibilidade |
| TEST_SIZE | 0.2 | 20% das interações mais recentes reservadas para teste (split temporal) |
| VALIDATION_SIZE | 0.2 | 20% das interações intermediárias reservadas para validação (split temporal) |
| RECOMMENDATION_K | 10 | Tamanho da lista de recomendações entregue ao usuário |

## Hiperparâmetros dos modelos

| Parâmetro | Modelo | Valor | Descrição |
| --- | --- | --- | --- |
| SVD_N_FACTORS | SVD | 50 | Número de fatores latentes |
| MLP_EMBEDDING_DIM | MLP | 64 | Dimensão dos embeddings de usuário e item |
| MLP_HIDDEN_DIMS | MLP | [128, 64] | Tamanho das camadas ocultas |
| MLP_EPOCHS | MLP | 50 | Número máximo de épocas (early stopping pode interromper antes) |
| MLP_LEARNING_RATE | MLP | 0.001 | Taxa de aprendizado |
| MLP_BATCH_SIZE | MLP | 256 | Tamanho do batch |

# Métricas de avaliação

O intuito é agregar no ticket médio geral do cliente que já comprou (Não de uma compra específica), utilizando o histórico de interações do mesmo para recomendar produtos que ele tem mais chances de comprar.

**interação principal (ground thruth):** "transaction"

## Métricas dos modelos:

| Métrica | O que mede | Interpretação |
| --- | --- | --- |
| Precision@K | Dos K recomendados, quantos o usuário interagiu (transaction) | 0 a 1, quanto maior melhor |
| Recall@K | Dos itens interagidos (transaction), quantos foram recomendados | 0 a 1, quanto maior melhor |
| NDCG@K | Qualidade da ordenação, penaliza acertos no final | 0 a 1, quanto maior melhor |
| Hit Rate@K | % de usuários com pelo menos 1 acerto nos K recomendados | 0 a 1, quanto maior melhor |

## Métricas de negócio:

| Métrica | O que mede | Usa value? | Interpretação |
| --- | --- | --- | --- |
| Coverage | % do catálogo recomendado | Não | 0 a 1, quanto maior melhor |
| Revenue@K | Soma do value dos itens recomendados que foram comprados | Sim | Quanto maior melhor (R$) |

## Hierarquia de métricas para seleção de modelo

| Papel | Métrica | Motivo |
| --- | --- | --- |
| Principal | NDCG@K | Penaliza acertos no final da lista — avalia qualidade da ordenação |
| Desempate | Hit Rate@K | Mais interpretável para negócio; desempata quando NDCG é próximo |
| Monitoramento | Precision@K e Recall@K | Complementares; revelam se o modelo é conservador ou abrangente |
| Negócio | Coverage e Revenue@K | Não entram na seleção — justificam o valor real do modelo |

**Avaliação em duas fases:**
1. Seleção de configuração (ex: melhor `n_factors` do SVD) → NDCG@K no `val_df`
2. Avaliação final → todas as 6 métricas no `test_df`, tocado apenas uma vez


# Design patterns que vamos utilizar

- Strategy (diferentes modelos de recomendação)
    - Organiza a estrutura dos modelos, garantindo que todos tenham a mesma interface (fit, recommend, get_params)

- Factory Method (RecommenderFactory)
    - Permite criar instâncias de modelos dinamicamente com base em uma string de configuração

# Estrutura de desenvolvimento

```
1. EDA (notebook 01)
        ↓
2. src/data/
   ├── loader.py
   ├── preprocessor.py   ← sklearn Pipeline
   └── split.py
        ↓
3. src/tracking/
   └── mlflow_utils.py   ← configure_mlflow_tracking, build_experiment_tags
        ↓
4. src/models/
   ├── base.py           ← BaseRecommender
   ├── factory.py        ← RecommenderFactory
   ├── popularity.py     ← baseline pronto
   ├── mlp.py            ← stub
   └── svd.py  ← stub
        ↓
5. notebooks/02_experiments.ipynb  ← só aqui você experimenta
        ↓
6. Preenche os stubs (mlp.py, svd.py) com base no que aprendeu no notebook
        ↓
7. src/training/train.py  ← formaliza o que o notebook fez
        ↓
8. DVC pipeline + Docker
```
_*stub é uma classe vazia só para preencher depois_

**Por que essa ordem**
1. **EDA primeiro** — você não escreve código de processamento sem entender os dados. O loader, o preprocessor e o split dependem do formato real do dataset (colunas, tipos, distribuição de interações, se tem timestamp, etc.).

2. **src/data/ antes do modelo** — o modelo precisa receber dados limpos e no formato certo. Sem o preprocessor pronto, qualquer experimento no notebook vai ser manual e irreprodutível.

3. **MLflow antes dos modelos** — se você implementa os modelos antes, vai treinar sem rastrear. Quando for conectar o MLflow depois, vai ter que refatorar o código de treino. Configurar antes custa pouco e garante que o notebook já rastreia desde o primeiro experimento.

4. **Design patterns antes do notebook de experimentos** — o notebook precisa importar os modelos e trocar entre eles com facilidade. Sem a Factory, você escreve código diferente pra cada modelo no notebook. Com a Factory, uma linha instancia qualquer modelo. O notebook fica limpo e os experimentos são comparáveis.

5. **Notebook de experimentos antes de implementar MLP e MF** — você usa o notebook pra entender o que o modelo precisa: qual formato de entrada, quantas épocas convergem, quais hiperparâmetros importam. Implementar o MLP no escuro antes disso é arriscado.

6. **train.py depois do notebook** — o train.py é a versão de produção do que você já validou no notebook. Formalizar antes de validar significa reescrever.

7. **DVC e Docker por último** — são camadas de infraestrutura em cima de código que já funciona. Containerizar código quebrado só esconde o problema.

**Por que Strategy:**
O enunciado exige comparar MLP com baselines usando ≥ 4 métricas. Isso significa que você vai ter 3 modelos fazendo a mesma coisa — receber interações, treinar, recomendar. A variação é o algoritmo, não o processo em volta. Strategy é exatamente o padrão para isso: define uma interface comum, cada modelo implementa do seu jeito, o código de treino e avaliação não muda.

Sem Strategy, cada arquivo que usa um modelo precisa de if/elif por tipo de modelo. Com Strategy, nenhum arquivo sabe qual modelo está usando — só chama fit() e recommend().

### Sem strategy
```python
# src/training/train.py
if params["model"] == "mlp":
    model = MLPRecommender(embedding_dim=64, hidden_dims=[128, 64])
    model.fit_pytorch(train_data, epochs=50)
    preds = model.forward(users_val)
elif params["model"] == "svd":
    model = SVDRecommender(n_factors=50)
    model.fit_sklearn(train_data)
    preds = model.transform(users_val)
elif params["model"] == "popularity":
    model = PopularityRecommender()
    model.count_items(train_data)
    preds = model.get_top_items()
```
```python
# src/evaluation/evaluate.py — mesmo problema aqui
if model_type == "mlp":
    preds = model.forward(users)
elif model_type == "svd":
    preds = model.transform(users)
elif model_type == "popularity":
    preds = model.get_top_items()

# adicionar um novo modelo = abrir TODOS os arquivos e adicionar elif
```
### Com strategy
```python
# src/models/base.py
class BaseRecommender(ABC):
    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> None: ...

    @abstractmethod
    def recommend(self, user_id: int, k: int) -> list[int]: ...

    @abstractmethod
    def get_params(self) -> dict: ...
```
```python
# src/training/train.py — não sabe qual modelo é
model.fit(train_data)         # funciona pra qualquer um
preds = model.recommend(user_id, k=10)  # igual pra todos

# src/evaluation/evaluate.py — igual, sem if/elif
preds = model.recommend(user_id, k=10)
metrics = evaluate(preds, ground_truth)

```

**Por que Factory:**
O DVC parametriza o pipeline via params.yaml. Você precisa conseguir trocar o modelo mudando uma linha no arquivo de configuração, sem tocar no código. A Factory lê o tipo do params, instancia o objeto certo e devolve. Sem ela, o train.py precisa de if/elif pra decidir qual modelo criar — e isso quebra toda vez que você adiciona um novo modelo.

### Sem factory
```python
# train.py
if params["model"] == "mlp":
    model = MLPRecommender(
        embedding_dim=params["embedding_dim"],
        hidden_dims=params["hidden_dims"]
    )
elif params["model"] == "svd":
    model = SVDRecommender(n_factors=params["n_factors"])
elif params["model"] == "popularity":
    model = PopularityRecommender()
# novo modelo no futuro = abre train.py de novo
```
### Com factory
```python
# src/models/factory.py
class RecommenderFactory:
    _registry = {
        "mlp": MLPRecommender,
        "svd": SVDRecommender,
        "popularity": PopularityRecommender,
    }

    @classmethod
    def create(cls, name: str, config: dict) -> BaseRecommender:
        if name not in cls._registry:
            raise ValueError(f"Modelo desconhecido: {name}")
        return cls._registry[name](config)
```
```python
# train.py — uma linha, independente de qual modelo
model = RecommenderFactory.create(params["model"], params)

# params.yaml — trocar modelo = mudar essa linha
model:
  type: "svd"   # era "mlp", virou "svd", train.py não mudou
```
