# Datasets que podemos utilizar

- https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis
- https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset *← Claude mandou escolher esse pelo realismo e volumetria*
- https://www.kaggle.com/datasets/sherinclaudia/movielens



# Modelos que podemos utilizar

|Modelo | Biblioteca |
| --- | --- |
|MLP embedding-based | PyTorch
|PopularityRecommender | Lógica própria
|SVD/NMF | surprise


# Design patterns que vamos utilizar

- Factory Method (RecommenderFactory)
- Strategy (diferentes modelos de recomendação)


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
   └── mlflow_utils.py   ← configure_mlflow_tracking, build_default_run_tags
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
