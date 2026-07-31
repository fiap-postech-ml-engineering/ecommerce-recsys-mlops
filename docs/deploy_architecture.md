# Arquitetura de Deploy

## 🗺️ Visão geral

Microsserviço stateless exposto via API REST síncrona, implementada com **FastAPI + Uvicorn**, containerizado via **Docker** (build multi-stage) e orquestrado localmente via **docker-compose**. O serviço carrega o modelo em `Production` do MLflow Model Registry e encapsula a inferência de recomendação sem exigir que o cliente conheça os detalhes do modelo.

A API expõe os seguintes endpoints (ver `src/api/routes/`):

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Informações da API e status do serviço |
| `GET` | `/health` | Health check — status da API e se o modelo está carregado (`recommender_service.is_ready`) |
| `POST` | `/recommend` | Recomendação de top-K produtos para um `user_id` |

A documentação interativa é gerada automaticamente pelo FastAPI e pode ser acessada em `/docs` (Swagger) ou `/redoc` quando a API estiver rodando.

## 🤔 O porquê da escolha dessa arquitetura

FastAPI foi escolhido pela validação automática de payload via Pydantic, performance assíncrona nativa e integração direta com o ecossistema Python usado no restante do pipeline (PyTorch, MLflow, DVC).

- Latência medida do `/recommend` (ItemKNN, 30 requisições locais, modelo já carregado em memória): p50 ≈ 3,8ms, p95 ≈ 5,8ms, p99 ≈ 10,9ms. Medição local (loopback, sem rede real) — serve como piso, não como SLA de produção.
- O modelo é carregado uma única vez no startup (`lifespan`, ver `src/api/app.py`) e mantido em memória, evitando I/O repetido do MLflow Registry a cada request.

## 🔁 Qual o funcionamento da API

1. **Startup (`lifespan`)** — `get_settings()` carrega a configuração e `recommender_service.load()` busca o modelo com alias `production` no MLflow Registry (`MLFLOW_MODEL_NAME`, `MLFLOW_MODEL_ALIAS` em `src/config.py`).
2. **Middleware de observabilidade** — gera ou propaga o header `X-Request-ID` (UUID), mede a latência da requisição e loga `request.completed` (método, path, status, latência, IP) ao final; emite `WARNING` se a latência ultrapassar `LATENCY_WARN_MS` (ver `src/api/middleware.py`).
3. **Validação Pydantic** — `RecommendRequest` valida `user_id` e `k` (payload inválido retorna `422`, ver `tests/test_api.py`).
4. **Checagem de disponibilidade** — se o modelo não estiver carregado (`is_ready=False`), a API retorna `503` antes de tentar inferir (degradação graciosa, ver `src/api/routes/recommend.py`).
5. **Inferência** — `recommender_service.recommend(user_id, k)` delega ao modelo carregado via `BaseRecommender.recommend()` (Strategy).
6. **Resposta** — `RecommendResponse` com `user_id` e lista ordenada de `item_id` recomendados.

## ⚙️ Configurações e ambiente

Configuração via `.env` / `Pydantic Settings` (`src/config.py`), sem valores hardcoded nas rotas:

| Variável | Descrição |
|---|---|
| `APP_ENV` | Ambiente da aplicação (`development`/`production`, etc.) |
| `MODEL_VERSION` | Versão exposta no título/health da API |
| `RECOMMENDATION_K` | Tamanho padrão da lista de recomendações |
| `MLFLOW_TRACKING_URI` | URI de tracking do MLflow |
| `MLFLOW_MODEL_NAME` | Nome do modelo no Registry (`workspace.default.ecomm-recsys-itemknn`) |
| `MLFLOW_MODEL_ALIAS` | Alias do estágio a carregar (`production`) |
| `LATENCY_WARN_MS` | Limiar de latência para log em nível `WARNING` |

## 🐳 Containerização

- **`Dockerfile`** — multi-stage: `base` (Python 3.13-slim + `uv`) → `builder` (`uv sync --frozen --no-dev`) → `runtime` (usuário não-root, apenas `.venv` resolvido + código necessário).
- **`docker-compose.yml`** — serviço `app`, build a partir do `Dockerfile` (`target: runtime`), bind mounts para hot-reload em dev, healthcheck via `GET /`, porta `8000`.
- Comandos: `make docker-build`, `make docker-up[-detached]`, `make docker-down`/`make stop`, `make docker-logs`, `make docker-check` (smoke test: sobe, valida `/`, derruba).

## ⚠️ Limitações atuais

| Limitação | Impacto |
|---|---|
| Sem CI/CD configurado | Testes, lint e deploy executados manualmente |
| Processo único (sem workers paralelos) | Sem paralelismo real de CPU em picos de carga |
| Sem autoscaling | Gargalo sob alta demanda |
| Sem versionamento de endpoint (`/v1/`) | Breaking changes afetam todos os clientes |
| Sem autenticação | API aberta, sem controle de acesso |
| Sem rate limiting | Vulnerável a abuso ou sobrecarga acidental |
| Modelo carregado só no startup (`lifespan`) | Promover um modelo novo em Production no MLflow Registry não afeta uma API já em execução — exige reiniciar o processo para recarregar |
