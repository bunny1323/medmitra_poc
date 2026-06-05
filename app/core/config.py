import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    APP_NAME: str = "medmitra-ml-search"
    APP_ENV: str = "development"
    API_V1_PREFIX: str = "/api/v1"
    INTERNAL_API_KEY: str = "medmitra_internal_secure_key_123"

    # Qdrant
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "medmitra_knowledge"
    QDRANT_COLLECTION_PREFIX: str = "medmitra_knowledge"
    QDRANT_COLLECTION_ALIAS: str = "medmitra_knowledge_current"
    QDRANT_HOST_PORT: int = 6333

    # Embedding / MedCPT
    EMBEDDING_MODEL: str = "NeuML/pubmedbert-base-embeddings"
    EMBEDDING_DEVICE: str = "cpu"
    DENSE_VECTOR_SIZE: int = 768
    EMBEDDING_BATCH_SIZE: int = 16
    HF_TOKEN: Optional[str] = None

    # MedCPT specific
    MEDCPT_QUERY_MODEL: str = "ncbi/MedCPT-Query-Encoder"
    MEDCPT_ARTICLE_MODEL: str = "ncbi/MedCPT-Article-Encoder"

    # Chunking (kept for backward compat with book ingestion)
    CHUNK_TARGET_TOKENS: int = 260
    CHUNK_MIN_TOKENS: int = 70
    CHUNK_MAX_TOKENS: int = 360
    CHUNK_OVERLAP_TOKENS: int = 45

    # Retrieval / RRF
    RETRIEVAL_DENSE_PREFETCH_K: int = 18
    RETRIEVAL_SPARSE_PREFETCH_K: int = 18
    RETRIEVAL_FINAL_K: int = 4
    RETRIEVAL_MIN_RELEVANCE: float = 0.30
    DENSE_TOP_K: int = 15
    SPARSE_TOP_K: int = 15
    FINAL_TOP_K: int = 5
    RRF_K: int = 60

    # LLM (Groq by default — Qwen is a future local option)
    LLM_PROVIDER: str = "groq"
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_API_KEY: Optional[str] = None
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 500
    LLM_TIMEOUT_SECONDS: float = 30.0

    # Upload
    MAX_UPLOAD_MB: int = 10
    LOG_LEVEL: str = "INFO"

    # Search relevance thresholds
    DISEASE_SEARCH_MIN_RELEVANCE: float = 0.10
    MEDICINE_SEARCH_MIN_RELEVANCE: float = 0.10
    USE_DEMO_SEED_DATA: bool = True

    # Kaggle credentials (optional — for download automation)
    KAGGLE_USERNAME: Optional[str] = None
    KAGGLE_KEY: Optional[str] = None

    # Agent
    AGENT_ENABLED: bool = True
    LLM_ENABLED: bool = True

    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # --- Path properties ---
    @property
    def books_dir(self) -> str:
        return os.path.join(self.BASE_DIR, "app", "data", "books")

    @property
    def reports_dir(self) -> str:
        return os.path.join(self.BASE_DIR, "app", "data", "reports")

    @property
    def curated_dir(self) -> str:
        return os.path.join(self.BASE_DIR, "app", "data", "curated")

    @property
    def emergency_rules_path(self) -> str:
        return os.path.join(self.BASE_DIR, "app", "data", "emergency_rules.json")

    @property
    def curated_emergency_rules_path(self) -> str:
        return os.path.join(self.curated_dir, "emergency_rules.json")

    @property
    def processed_chunks_path(self) -> str:
        return os.path.join(self.BASE_DIR, "app", "data", "processed_chunks.json")

    @property
    def indexes_dir(self) -> str:
        return os.path.join(self.BASE_DIR, "app", "data", "indexes")

    @property
    def raw_data_dir(self) -> str:
        return os.path.join(self.BASE_DIR, "app", "data", "raw")

    @property
    def processed_data_dir(self) -> str:
        return os.path.join(self.BASE_DIR, "app", "data", "processed")

    @property
    def processed_diseases_path(self) -> str:
        return os.path.join(self.processed_data_dir, "diseases.json")

    @property
    def processed_medicines_path(self) -> str:
        return os.path.join(self.processed_data_dir, "medicines.json")

    @property
    def symptom_queries_path(self) -> str:
        return os.path.join(self.processed_data_dir, "symptom_queries.json")

    # Legacy demo paths (kept for backward compat)
    @property
    def disease_demo_path(self) -> str:
        return os.path.join(self.BASE_DIR, "app", "data", "demo", "disease_records.json")

    @property
    def medicine_demo_path(self) -> str:
        return os.path.join(self.BASE_DIR, "app", "data", "demo", "medicine_records.json")

    @property
    def medcpt_embeddings_path(self) -> str:
        return os.path.join(self.indexes_dir, "medcpt_embeddings.npy")

    @property
    def chunk_metadata_path(self) -> str:
        return os.path.join(self.indexes_dir, "chunk_metadata.json")

    @property
    def bm25_index_path(self) -> str:
        return os.path.join(self.indexes_dir, "bm25_index.pkl")

    @property
    def disease_embeddings_path(self) -> str:
        return os.path.join(self.indexes_dir, "disease_embeddings.npy")

    @property
    def medicine_embeddings_path(self) -> str:
        return os.path.join(self.indexes_dir, "medicine_embeddings.npy")

    @property
    def disease_bm25_path(self) -> str:
        return os.path.join(self.indexes_dir, "disease_bm25.pkl")

    @property
    def medicine_bm25_path(self) -> str:
        return os.path.join(self.indexes_dir, "medicine_bm25.pkl")


settings = Settings()
