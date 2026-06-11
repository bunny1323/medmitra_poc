from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"


# ---------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True,
)


# ---------------------------------------------------------------------
# API keys and security
# ---------------------------------------------------------------------

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


# ---------------------------------------------------------------------
# Qdrant database settings
# ---------------------------------------------------------------------

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "medical_books",
)
# Qdrant Cloud timeout
QDRANT_TIMEOUT_SECONDS = int(
    os.getenv("QDRANT_TIMEOUT_SECONDS", "120")
)

# Dense embedding dimensions for BGE-small
DENSE_VECTOR_SIZE = int(
    os.getenv("DENSE_VECTOR_SIZE", "384")
)

# Cloud-friendly upload settings
INGEST_MAX_RETRIES = int(
    os.getenv("INGEST_MAX_RETRIES", "10")
)

INGEST_RETRY_BASE_SECONDS = float(
    os.getenv("INGEST_RETRY_BASE_SECONDS", "3")
)

# Enable normal HNSW indexing after bulk upload
HNSW_M = int(
    os.getenv("HNSW_M", "16")
)

# ---------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------

LLAMA_MODEL = os.getenv(
    "LLAMA_MODEL",
    "llama-3.3-70b-versatile",
)

DENSE_MODEL_NAME = os.getenv(
    "DENSE_MODEL_NAME",
    "BAAI/bge-small-en-v1.5",
)

SPARSE_MODEL_NAME = os.getenv(
    "SPARSE_MODEL_NAME",
    "prithivida/Splade_PP_en_v1",
)


# ---------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------

DENSE_PREFETCH_K = int(
    os.getenv("DENSE_PREFETCH_K", "20")
)

SPARSE_PREFETCH_K = int(
    os.getenv("SPARSE_PREFETCH_K", "20")
)

FINAL_TOP_K = int(
    os.getenv("FINAL_TOP_K", "5")
)


# ---------------------------------------------------------------------
# Ingestion settings
# ---------------------------------------------------------------------

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1000")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "200")
)

INGEST_BATCH_SIZE = int(
    os.getenv("INGEST_BATCH_SIZE", "15")
)


# ---------------------------------------------------------------------
# LLM generation settings
# ---------------------------------------------------------------------

LLM_TEMPERATURE = float(
    os.getenv("LLM_TEMPERATURE", "0.1")
)

LLM_MAX_TOKENS = int(
    os.getenv("LLM_MAX_TOKENS", "600")
)

LLM_TIMEOUT_SECONDS = int(
    os.getenv("LLM_TIMEOUT_SECONDS", "30")
)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")