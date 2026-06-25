from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="forbid",
    )

    # App
    APP_ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="text")
    LATENCY_WARN_MS: int = Field(default=1000, gt=0)
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    MODEL_VERSION: str = Field(default="0.1.0")

    # ML training
    RANDOM_SEED: int = Field(default=42)
    TEST_SIZE: float = Field(default=0.2, gt=0.0, lt=1.0)
    VALIDATION_SIZE: float = Field(default=0.2, gt=0.0, lt=1.0)
    RECOMMENDATION_K: int = Field(default=10, gt=0)

    # SVD
    SVD_N_FACTORS: int = Field(default=50, gt=0)

    # MLP
    MLP_EMBEDDING_DIM: int = Field(default=64, gt=0)
    MLP_HIDDEN_DIMS: list[int] = Field(default=[128, 64])
    MLP_EPOCHS: int = Field(default=50, gt=0)
    MLP_LEARNING_RATE: float = Field(default=0.001, gt=0.0)
    MLP_BATCH_SIZE: int = Field(default=256, gt=0)

    # MLflow / Databricks
    MLFLOW_TRACKING_URI: str = Field(default="databricks")
    MLFLOW_EXPERIMENT_NAME: str = Field(default="/Shared/mlflow_ecomm_recsys")
    DATABRICKS_HOST: str | None = Field(default=None)
    DATABRICKS_TOKEN: SecretStr | None = Field(default=None)

    # DVC / S3
    DVC_REMOTE_URL: str | None = Field(default=None)
    AWS_ACCESS_KEY_ID: SecretStr | None = Field(default=None)
    AWS_SECRET_ACCESS_KEY: SecretStr | None = Field(default=None)
    AWS_DEFAULT_REGION: str = Field(default="us-east-1")

    @field_validator("APP_ENV")
    @classmethod
    def _validate_app_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"APP_ENV must be one of {allowed}")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.upper()

    @field_validator("LOG_FORMAT")
    @classmethod
    def _validate_log_format(cls, v: str) -> str:
        allowed = {"json", "text"}
        if v.lower() not in allowed:
            raise ValueError(f"LOG_FORMAT must be one of {allowed}")
        return v.lower()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    from src.logging_config import setup_logging

    settings = Settings()
    setup_logging(settings)
    return settings
