"""
app/services/retrieval_service.py
──────────────────────────────────
Handles vector search against Qdrant using Hybrid Search for the medical book.
"""

from __future__ import annotations

import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Using fastembed for BOTH Dense and Sparse to avoid DLL blocks
from fastembed import TextEmbedding, SparseTextEmbedding

_DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"
_QDRANT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "qdrant_db")

MEDICAL_BOOK_COLLECTION = "medical_book_2025"

_dense_model: TextEmbedding | None = None
_sparse_model: SparseTextEmbedding | None = None
_qdrant_client: QdrantClient | None = None

def get_dense_model() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        print(f"[retrieval_service] Loading dense model: {_DENSE_MODEL_NAME}")
        _dense_model = TextEmbedding(_DENSE_MODEL_NAME)
    return _dense_model

def get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        print(f"[retrieval_service] Loading sparse model: {_SPARSE_MODEL_NAME}")
        _sparse_model = SparseTextEmbedding(_SPARSE_MODEL_NAME)
    return _sparse_model

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if qdrant_url and qdrant_api_key:
            _qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            _qdrant_client = QdrantClient(path=_QDRANT_PATH)
    return _qdrant_client

def collection_exists(collection_name: str) -> bool:
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        return any(c.name == collection_name for c in collections)
    except Exception:
        return False

def search(
    query: str,
    top_k: int = 5,
) -> dict[str, Any]:
    collection_name = MEDICAL_BOOK_COLLECTION
    
    if not query or not query.strip():
        return _empty_result(query, "Empty query provided.")

    try:
        client = get_qdrant_client()
        if not collection_exists(collection_name):
             return _empty_result(query, f"Collection {collection_name} not found. Run ingest script.")
    except Exception as e:
        return _empty_result(query, f"Qdrant connection failed: {e}")

    dense_model = get_dense_model()
    # fastembed returns a generator, so we convert to list and take the first item
    dense_embedding = list(dense_model.embed([query.strip()]))[0].tolist()
    
    sparse_model = get_sparse_model()
    sparse_query = list(sparse_model.query_embed(query.strip()))[0]

    try:
        search_result = client.query_points(
            collection_name=collection_name,
            prefetch=[
                qmodels.Prefetch(
                    query=dense_embedding,
                    using="dense",
                    limit=20
                ),
                qmodels.Prefetch(
                    query=qmodels.SparseVector(
                        indices=sparse_query.indices.tolist(),
                        values=sparse_query.values.tolist()
                    ),
                    using="sparse",
                    limit=20
                )
            ],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=top_k
        ).points
    except Exception as e:
        return _empty_result(query, f"Qdrant query failed: {e}")

    results = []
    for rank, point in enumerate(search_result, start=1):
        score = getattr(point, "score", 0.0)
        results.append({
            "rank": rank,
            "id": str(point.id),
            "content": point.payload.get("content", ""),
            "relevance_score": round(score, 4),
            "metadata": point.payload or {},
        })

    top_relevance = results[0]["relevance_score"] if results else 0.0

    return {
        "query": query,
        "results": results,
        "top_relevance": top_relevance,
        "model": "Hybrid (BGE-small + SPLADE)"
    }

def _empty_result(query: str, reason: str) -> dict:
    return {
        "query": query,
        "results": [],
        "top_relevance": 0.0,
        "error": reason
    }
