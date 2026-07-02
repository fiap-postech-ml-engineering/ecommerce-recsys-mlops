FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/


FROM base AS builder

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev


FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN addgroup --system app \
    && adduser --system --ingroup app app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY pyproject.toml README.md LICENSE ./

RUN mkdir -p /app/data /app/models /app/logs \
    && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]