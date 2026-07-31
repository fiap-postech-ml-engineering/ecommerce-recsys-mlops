"""Valida o ambiente antes de rodar o pipeline (TCF1-159).

Confere que `.env`/variáveis de ambiente carregam em `Settings` sem erro e que as
integrações externas usadas pelo pipeline (MLflow/Databricks, DVC remote em S3) têm
credenciais configuradas. Falha cedo com mensagem clara em vez de deixar o erro
estourar no meio de um `dvc repro` ou de um treino longo.

Uso: `python -m scripts.validate_env` (ou `uv run python -m scripts.validate_env`).
"""

import logging
import sys

from pydantic import ValidationError

from src.config import Settings

logger = logging.getLogger(__name__)

PLACEHOLDER_VALUE = "__set_me__"

# Cada grupo só é obrigatório se a integração correspondente for usada; ausência não
# impede o boot da aplicação (todo campo de Settings tem default), mas quebra o
# pipeline mais adiante sem essas credenciais.
DATABRICKS_VARS = ("DATABRICKS_HOST", "DATABRICKS_TOKEN")
DVC_REMOTE_VARS = ("DVC_REMOTE_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")


def _load_settings() -> Settings:
    """Instancia `Settings`, deixando a `ValidationError` do Pydantic propagar."""
    return Settings()


def _missing_or_placeholder(settings: Settings, var_names: tuple[str, ...]) -> list[str]:
    """Retorna os nomes de `var_names` ausentes ou ainda com o placeholder do `.env.example`.

    Args:
        settings: Instância de `Settings` já carregada.
        var_names: Nomes dos campos de `Settings` a conferir.

    Returns:
        Lista de nomes de variáveis não configuradas (vazia se todas estiverem ok).
    """
    problems = []
    for name in var_names:
        value = getattr(settings, name)
        raw = value.get_secret_value() if hasattr(value, "get_secret_value") else value
        if raw is None or raw == PLACEHOLDER_VALUE:
            problems.append(name)
    return problems


def main() -> None:
    """Ponto de entrada CLI: valida `Settings` e credenciais de integrações externas."""
    try:
        settings = _load_settings()
    except ValidationError as exc:
        logger.error("Settings inválido — corrija o .env:\n%s", exc)
        sys.exit(1)

    logger.info("Settings carregado com sucesso (APP_ENV=%s).", settings.APP_ENV)

    missing_databricks: list[str] = []
    if settings.MLFLOW_TRACKING_URI == "databricks":
        missing_databricks = _missing_or_placeholder(settings, DATABRICKS_VARS)
        if missing_databricks:
            logger.warning(
                "MLFLOW_TRACKING_URI=databricks mas faltam credenciais: %s. "
                "Tracking/Registry do MLflow vai falhar até isso ser preenchido no .env.",
                ", ".join(missing_databricks),
            )
    else:
        logger.info(
            "MLFLOW_TRACKING_URI=%s — MLflow local, sem credenciais necessárias.",
            settings.MLFLOW_TRACKING_URI,
        )

    missing_dvc = _missing_or_placeholder(settings, DVC_REMOTE_VARS)
    if missing_dvc:
        logger.warning(
            "Credenciais do DVC remote (S3) não configuradas: %s. "
            "`dvc push`/`dvc pull` vão falhar até isso ser preenchido no .env.",
            ", ".join(missing_dvc),
        )

    if missing_databricks or missing_dvc:
        logger.warning("Ambiente validado com avisos — ver mensagens acima.")
    else:
        logger.info("Ambiente validado sem avisos.")


if __name__ == "__main__":
    main()
