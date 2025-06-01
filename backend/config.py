
from pydantic import BaseSettings

class Settings(BaseSettings):
    port: int = 8000
    embed_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "google/flan-t5-small"
    max_length: int = 200
    log_level: str = "INFO"

    class Config:
        env_file = ".env"   
        env_file_encoding = "utf-8"

settings = Settings()
