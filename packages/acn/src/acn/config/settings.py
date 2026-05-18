from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ACN_",
        extra="ignore",
        case_sensitive=False,
    )

    env: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_allow_origin_regex: str | None = r"https?://(localhost|127\.0\.0\.1):\d+"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "acn"
    postgres_user: str = "acn"
    postgres_password: str = Field(default="acn", repr=False)

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_root: str = "s3://mlflow/"
    mlflow_experiment_name: str = "ACN Visual Adaptive Demo"

    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = Field(default="minioadmin", repr=False)
    minio_secret_key: str = Field(default="minioadmin", repr=False)
    minio_bucket: str = "mlflow"
    minio_artifact_bucket: str = "acn-artifacts"
    minio_region: str = "us-east-1"

    dashboard_telemetry_path: Path | None = None
    runtime_stack_enabled: bool = False

    @property
    def database_url(self) -> str:
        return (
            "postgresql+psycopg://"
            f"{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> RedisDsn:
        return RedisDsn.build(
            scheme="redis",
            host=self.redis_host,
            port=self.redis_port,
            path=f"{self.redis_db}",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
