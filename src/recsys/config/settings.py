from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações do projeto, carregadas das variáveis de ambiente ou .env.

    Attributes:
        environment (str): Ambiente de execução (ex: development, production).
        log_level (str): Nível de log (ex: info, debug).
        mlflow_tracking_uri (str): URI do servidor MLflow.
        data_raw_dir (Path): Diretório para dados brutos.
        data_processed_dir (Path): Diretório para dados processados.
        models_dir (Path): Diretório para modelos treinados.
        random_seed (int): Semente aleatória para reprodutibilidade.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: str = "development"
    log_level: str = "INFO"

    mlflow_tracking_uri: str = "http://localhost:5000"

    data_raw_dir: Path = Path("data/raw")
    data_processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    random_seed: int = 42


@lru_cache
def get_settings() -> Settings:
    """Retorna uma instância de Settings, carregando as configurações do ambiente.

    Returns:
        Settings: Instância de configurações do projeto.
    """
    return Settings()
