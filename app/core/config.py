from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "qwen3:4b"

    qdrant_path: str = "qdrant_data"
    qdrant_collection: str = "enterprise_documents"

    reranker_min_score: float = 0.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    ollama_base_url: str = "http://localhost:11434"


settings = Settings()
