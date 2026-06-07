import torch
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración centralizada por entorno (desarrollo / producción)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: Literal["development", "production"] = "development"

    model_path: str = "models/resnet18_best.pth"
    label_encoder_path: str = "models/label_encoder.pkl"
    data_dir: str = "data/fused"
    raw_data_dir: str = "data/raw"
    metadata_path: str = "data/metadata/fall_dataset_fused_metadata.csv"
    mlruns_dir: str = "mlruns"

    image_size: int = 224
    num_epochs: int = 5
    num_epochs_head: int = 3
    num_epochs_finetune: int = 12
    batch_size: int = 32
    learning_rate: float = 0.001
    finetune_learning_rate: float = 0.0001
    early_stopping_patience: int = 4
    positive_class: str = "fall"

    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_url: str = "http://localhost:8080"

    roboflow_api_key: str | None = None

    database_url: str | None = None
    high_risk_min_falls_week: int = 2

    @property
    def effective_database_url(self) -> str | None:
        if self.database_url:
            return self.database_url
        if self.app_env == "development":
            return "sqlite:///./data/analytics/fall_analytics.db"
        return None

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

MODEL_PATH = settings.model_path
LABEL_ENCODER_PATH = settings.label_encoder_path
DEVICE = settings.device
