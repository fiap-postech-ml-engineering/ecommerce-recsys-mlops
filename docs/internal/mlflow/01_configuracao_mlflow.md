# Configuração do MLflow (Databricks)

Por padrão, o projeto usa o **Databricks** como backend de tracking do MLflow
(`MLFLOW_TRACKING_URI=databricks` em `src/config.py`). Este documento explica como
obter as credenciais, configurar o `.env` local e validar que o tracking está
funcionando.

## 1. Configurar o `.env`

Duplique o `.env.example` para `.env` (se ainda não existir):

```bash
cp .env.example .env
```

Edite o `.env` e preencha as variáveis da seção `=== MLflow / Databricks ===`:

```dotenv
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=https://dbc-xxxxxxxx-xxxx.cloud.databricks.com/
DATABRICKS_TOKEN=dapiXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
MLFLOW_EXPERIMENT_NAME="02 - ECOMM_RECSYS - Production"
```

- `DATABRICKS_HOST` e `DATABRICKS_TOKEN` vêm do passo 2.
- `MLFLOW_EXPERIMENT_NAME` é o nome do experimento compartilhado no workspace do
  Databricks — manter o valor padrão de `.env.example`, a não ser que o time decida
  criar um experimento próprio.

O `.env` está no `.gitignore` — nunca commitar token real.

## 2. Obter host e token do Databricks

1. Crie (ou acesse) uma conta no [Databricks Community Edition](https://community.cloud.databricks.com/)
   — gratuita, suficiente para o tracking de experimentos do MLflow.
   - Você precisa ter acesso ao nosso workspace compartilhado, peça acesso se ainda não tiver.
2. **Host**: após o login, copie a URL do workspace na barra de endereços, por
   exemplo `https://dbc-xxxxxxxx-xxxx.cloud.databricks.com/`. Esse valor vai em
   `DATABRICKS_HOST` no seu `.env`.
3. **Token**: no canto superior direito, clique no ícone de usuário →
   **Settings** → **Developer** → **Access tokens** → **Manage** → **Generate new
   token**. Dê um nome (ex: `ecommerce-recsys-mlops`), defina uma validade (ou
   deixe sem expiração), selecione "all APIs" em **API scope(s)** e copie o token — ele só é exibido uma vez. Esse
   valor vai em `DATABRICKS_TOKEN` no seu `.env`.


## 3. Testar a conexão

`src/tracking/mlflow_utils.configure_mlflow_tracking()` lê essas variáveis via
`Settings` (`src/config.py`) e configura o MLflow para apontar para o Databricks
automaticamente, sem precisar de argumentos.

### Teste automatizado

```bash
make test-slow
```

Esse comando roda os testes marcados como `slow`/`integration`, incluindo
`tests/test_tracking.py::test_configure_mlflow_tracking_authenticates_with_databricks`,
que autentica no Databricks com as credenciais do `.env` e lista os experimentos do
workspace. Se `DATABRICKS_HOST`/`DATABRICKS_TOKEN` não estiverem configurados, o
teste é pulado (`skip`).

### Teste manual (Python)

```python
from src.tracking import configure_mlflow_tracking
from mlflow.tracking import MlflowClient

configure_mlflow_tracking()

client = MlflowClient()
print(client.search_experiments(max_results=5))
```

Se a conexão estiver correta, o experimento configurado em
`MLFLOW_EXPERIMENT_NAME` aparece no workspace do Databricks, em **Experiments**.
