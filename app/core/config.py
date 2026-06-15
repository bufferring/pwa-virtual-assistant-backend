from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "LLM API"

    LLAMA_SERVER_URL: str = "http://host.containers.internal:8080"
    LLAMA_TIMEOUT: int = 300

    # Servicio de embeddings
    EMBEDDING_SERVER_URL: str = "http://host.containers.internal:8081"
    EMBEDDING_MODEL: str = "nomic-embed"

    # RAG
    CHROMA_PERSIST_DIR: str = "/app/chroma_data"
    CHROMA_COLLECTION_NAME: str = "unefa_knowledge"
    RAG_TOP_K: int = 4

    # Seguridad
    API_KEY: str | None = None

    # CORS
    CORS_ORIGINS: list[str] = ["*"]

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
