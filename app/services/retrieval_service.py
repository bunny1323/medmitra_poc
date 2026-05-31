"""
app/services/retrieval_service.py
──────────────────────────────────
Handles all vector search against persistent ChromaDB collections.

Design decisions:
  - Model is loaded once and cached (SentenceTransformer is slow to init)
  - Collections are opened read-only on each call (ChromaDB PersistentClient
    is safe to reopen multiple times — it just reads SQLite)
  - Distance is converted to a 0–1 relevance score for the UI
  - Relevance score ≠ diagnosis probability — always state this clearly

ChromaDB uses L2 distance by default; we request cosine distance by
setting the collection metadata at creation time (see build scripts).
Cosine distance range: 0 (identical) to 2 (opposite).
We map it to relevance = 1 - (distance / 2) so 1 = perfect, 0 = unrelated.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CHROMA_PATH   = _PROJECT_ROOT / "chroma_db"
_MODEL_NAME    = "NeuML/pubmedbert-base-embeddings"

# ── Collection names (must match the build scripts) ───────────────────────────

DISEASE_COLLECTION  = "medmitra_diseases"
MEDICINE_COLLECTION = "medmitra_medicines"

# ── Module-level singletons (loaded once per Streamlit process) ───────────────

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Return the shared SentenceTransformer model.
    Loads from HuggingFace cache on first call; subsequent calls are instant.
    """
    global _model
    if _model is None:
        print(f"[retrieval_service] Loading embedding model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
        print("[retrieval_service] Model loaded.")
    return _model


def _get_client() -> chromadb.PersistentClient:
    """Open (or reopen) the ChromaDB persistent client."""
    return chromadb.PersistentClient(path=str(_CHROMA_PATH))


def _cosine_distance_to_relevance(distance: float) -> float:
    """
    Convert ChromaDB cosine distance (0–2) to a relevance score (0–1).
    1.0 = perfect match, 0.0 = completely unrelated.
    """
    distance = max(0.0, min(2.0, distance))   # clamp to valid range
    return round(1.0 - (distance / 2.0), 4)


def collection_exists(collection_name: str) -> bool:
    """Return True if the named collection exists and has at least one record."""
    try:
        client = _get_client()
        col = client.get_collection(name=collection_name)
        return col.count() > 0
    except Exception:
        return False


def search(
    query: str,
    collection_name: str,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Embed the query and retrieve the top_k most similar documents.

    Parameters
    ----------
    query           : str   — user's natural-language question
    collection_name : str   — "medmitra_diseases" or "medmitra_medicines"
    top_k           : int   — number of results to return

    Returns
    -------
    {
        "query": str,
        "collection": str,
        "results": [
            {
                "rank": int,
                "id": str,
                "content": str,
                "distance": float,
                "relevance_score": float,   # 0–1; NOT diagnosis probability
                "metadata": dict,
            },
            ...
        ],
        "top_relevance": float,
        "model": str,
        "warning": str,   # always present — reminds user this is NOT diagnosis
    }
    """
    if not query or not query.strip():
        return _empty_result(query, collection_name, "Empty query provided.")

    # 1. Embed the query
    model = get_model()
    query_embedding = model.encode(
        query.strip(),
        normalize_embeddings=True,
        
    ).tolist()

    # 2. Open collection
    try:
        client = _get_client()
        col = client.get_collection(name=collection_name)
    except Exception as e:
        return _empty_result(
            query,
            collection_name,
            f"Collection '{collection_name}' not found. "
            f"Run the build script first.\n\nError: {e}",
        )

    # 3. Query
    try:
        raw = col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, col.count()),
            include=["documents", "distances", "metadatas"],
        )
    except Exception as e:
        return _empty_result(query, collection_name, f"ChromaDB query failed: {e}")

    # 4. Parse results
    results = []
    ids       = raw.get("ids",       [[]])[0]
    documents = raw.get("documents", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]

    for rank, (doc_id, content, dist, meta) in enumerate(
        zip(ids, documents, distances, metadatas), start=1
    ):
        relevance = _cosine_distance_to_relevance(dist)
        results.append({
            "rank":            rank,
            "id":              doc_id,
            "content":         content,
            "distance":        round(dist, 4),
            "relevance_score": relevance,
            "metadata":        meta or {},
        })

    top_relevance = results[0]["relevance_score"] if results else 0.0

    return {
        "query":         query,
        "collection":    collection_name,
        "results":       results,
        "top_relevance": top_relevance,
        "model":         _MODEL_NAME,
        "warning": (
            "⚕️ Relevance scores indicate semantic similarity to retrieved records — "
            "they are NOT diagnosis probabilities and do NOT confirm any medical condition. "
            "Always consult a qualified healthcare professional."
        ),
    }


def _empty_result(query: str, collection: str, reason: str) -> dict:
    """Return a well-formed empty result with an explanatory reason."""
    return {
        "query":         query,
        "collection":    collection,
        "results":       [],
        "top_relevance": 0.0,
        "model":         _MODEL_NAME,
        "error":         reason,
        "warning": (
            "⚕️ Relevance scores are NOT diagnosis probabilities. "
            "Always consult a qualified healthcare professional."
        ),
    }
