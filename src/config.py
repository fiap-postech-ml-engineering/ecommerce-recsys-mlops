from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
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
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    MODEL_VERSION: str = Field(default="0.1.0")

    # ML training
    RANDOM_SEED: int = Field(default=42)
    TEST_SIZE: float = Field(default=0.2, gt=0.0, lt=1.0)
    VALIDATION_SIZE: float = Field(default=0.2, gt=0.0, lt=1.0)

    # MLflow / Databricks
    MLFLOW_TRACKING_URI: str = Field(default="local")
    MLFLOW_EXPERIMENT_NAME: str = Field(default="tc2-recsys")
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

    @model_validator(mode="after")
    def _validate_databricks(self) -> "Settings":
        if self.MLFLOW_TRACKING_URI == "databricks":
            missing = [
                name
                for name, val in [
                    ("DATABRICKS_HOST", self.DATABRICKS_HOST),
                    ("DATABRICKS_TOKEN", self.DATABRICKS_TOKEN),
                ]
                if not val
            ]
            if missing:
                raise ValueError(f"MLFLOW_TRACKING_URI=databricks requires: {', '.join(missing)}")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
