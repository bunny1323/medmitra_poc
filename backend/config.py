from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    embedding_model: str = "all-MiniLM-L6-v2"

    chroma_persist_dir: str = "../data/vector_store"
    medicines_data_path: str = "../data/medicines.json"

    # Retrieval settings
    top_k_results: int = 3
    similarity_threshold: float = 0.50

    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000"
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()