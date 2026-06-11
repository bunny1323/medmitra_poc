"""
app/services/retrieval_service.py
──────────────────────────────────
Handles vector search against Qdrant using Hybrid Search for the medical books.
"""

from __future__ import annotations

from typing import Any

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core import config


# ---------------------------------------------------------------------
# Shared model and client instances
# ---------------------------------------------------------------------

_dense_model: TextEmbedding | None = None
_sparse_model: SparseTextEmbedding | None = None
_qdrant_client: QdrantClient | None = None


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def to_python_list(value: Any) -> list:
    """
    Convert NumPy arrays, FastEmbed vector values or other iterable
    objects into plain Python lists.

    This prevents errors when mocked test vectors are already lists.
    """
    if hasattr(value, "tolist"):
        return value.tolist()

    return list(value)


# ---------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------

def get_dense_model() -> TextEmbedding:
    """
    Load the dense embedding model once and reuse it across requests.
    """
    global _dense_model

    if _dense_model is None:
        print(
            "[retrieval_service] Loading dense model: "
            f"{config.DENSE_MODEL_NAME}"
        )

        _dense_model = TextEmbedding(
            model_name=config.DENSE_MODEL_NAME,
        )

    return _dense_model


def get_sparse_model() -> SparseTextEmbedding:
    """
    Load the sparse embedding model once and reuse it across requests.
    """
    global _sparse_model

    if _sparse_model is None:
        print(
            "[retrieval_service] Loading sparse model: "
            f"{config.SPARSE_MODEL_NAME}"
        )

        _sparse_model = SparseTextEmbedding(
            model_name=config.SPARSE_MODEL_NAME,
        )

    return _sparse_model


# ---------------------------------------------------------------------
# Qdrant client handling
# ---------------------------------------------------------------------

def get_qdrant_client() -> QdrantClient:
    """
    Return a cached Qdrant client.

    Cloud mode is used only when both QDRANT_URL and QDRANT_API_KEY
    are configured. Otherwise, local Qdrant storage is used.
    """
    global _qdrant_client

    if _qdrant_client is None:
        if config.QDRANT_URL and config.QDRANT_API_KEY:
            print("[retrieval_service] Connecting to Qdrant Cloud...")

            _qdrant_client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY,
                timeout = config.QDRANT_TIMEOUT_SECONDS,
            )

        else:
            local_path = config.BASE_DIR / "qdrant_db"

            print(
                "[retrieval_service] Using local Qdrant storage: "
                f"{local_path}"
            )

            _qdrant_client = QdrantClient(
                path=str(local_path),
            )

    return _qdrant_client


def close_qdrant_client() -> None:
    """
    Close the cached Qdrant client explicitly.

    Use this at the end of CLI scripts to avoid shutdown warnings.
    """
    global _qdrant_client

    if _qdrant_client is not None:
        try:
            _qdrant_client.close()

        finally:
            _qdrant_client = None


def collection_exists(collection_name: str) -> bool:
    """
    Check whether the configured Qdrant collection exists.
    """
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections

        return any(
            collection.name == collection_name
            for collection in collections
        )

    except Exception:
        return False


# ---------------------------------------------------------------------
# Hybrid dense + sparse retrieval
# ---------------------------------------------------------------------

def search(
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Search the medical knowledge base using:

    Dense retrieval:
        BAAI/bge-small-en-v1.5

    Sparse retrieval:
        prithivida/Splade_PP_en_v1
        or another configured sparse model

    Fusion:
        Reciprocal Rank Fusion (RRF)

    Returns a backend-friendly dictionary.
    """
    collection_name = config.QDRANT_COLLECTION

    if not query or not query.strip():
        return _empty_result(
            query=query,
            reason="Empty query provided.",
        )

    normalized_query = query.strip()

    try:
        client = get_qdrant_client()

        if not collection_exists(collection_name):
            return _empty_result(
                query=query,
                reason=(
                    f"Collection '{collection_name}' was not found. "
                    "Run the ingestion script first."
                ),
            )

    except Exception as exc:
        return _empty_result(
            query=query,
            reason=f"Qdrant connection failed: {exc}",
        )

    try:
        # -------------------------------------------------------------
        # Dense query embedding
        # -------------------------------------------------------------

        dense_model = get_dense_model()

        dense_result = list(
            dense_model.embed([normalized_query])
        )[0]

        dense_embedding = to_python_list(
            dense_result
        )

        # -------------------------------------------------------------
        # Sparse query embedding
        # -------------------------------------------------------------

        sparse_model = get_sparse_model()

        sparse_query = list(
            sparse_model.query_embed(normalized_query)
        )[0]

        sparse_indices = to_python_list(
            sparse_query.indices
        )

        sparse_values = to_python_list(
            sparse_query.values
        )

        # -------------------------------------------------------------
        # Qdrant hybrid query with RRF fusion
        # -------------------------------------------------------------

        query_response = client.query_points(
            collection_name=collection_name,
            prefetch=[
                qmodels.Prefetch(
                    query=dense_embedding,
                    using="dense",
                    limit=config.DENSE_PREFETCH_K,
                ),
                qmodels.Prefetch(
                    query=qmodels.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values,
                    ),
                    using="sparse",
                    limit=config.SPARSE_PREFETCH_K,
                ),
            ],
            query=qmodels.FusionQuery(
                fusion=qmodels.Fusion.RRF,
            ),
            limit=top_k,
        )

        search_points = query_response.points

    except Exception as exc:
        return _empty_result(
            query=query,
            reason=f"Qdrant query failed: {exc}",
        )

    # -----------------------------------------------------------------
    # Format results
    # -----------------------------------------------------------------

    results: list[dict[str, Any]] = []

    for rank, point in enumerate(search_points, start=1):
        payload = point.payload or {}
        score = float(getattr(point, "score", 0.0))

        results.append(
            {
                "rank": rank,
                "id": str(point.id),
                "content": payload.get("content", ""),
                "relevance_score": round(score, 4),
                "metadata": payload,
            }
        )

    top_relevance = (
        results[0]["relevance_score"]
        if results
        else 0.0
    )

    return {
        "query": query,
        "results": results,
        "top_relevance": top_relevance,
        "model": (
            "Hybrid "
            f"({config.DENSE_MODEL_NAME} "
            f"+ {config.SPARSE_MODEL_NAME} "
            "+ RRF)"
        ),
        "error": None,
    }


# ---------------------------------------------------------------------
# Empty-result fallback
# ---------------------------------------------------------------------

def _empty_result(
    query: str,
    reason: str,
) -> dict[str, Any]:
    """
    Return a consistent response when retrieval fails or finds nothing.
    """
    return {
        "query": query,
        "results": [],
        "top_relevance": 0.0,
        "model": (
            "Hybrid "
            f"({config.DENSE_MODEL_NAME} "
            f"+ {config.SPARSE_MODEL_NAME} "
            "+ RRF)"
        ),
        "error": reason,
    }