# backend/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    embed_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-small"
    port: int = 8000
    max_length: int = 200
    device: int = -1  # CPU
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

settings = Settings()