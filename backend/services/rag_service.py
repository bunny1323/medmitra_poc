import json
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from services.embedding_service import embed_query, embed_texts

_client: chromadb.ClientAPI | None = None
_collection = None
_initialized = False


def _resolve_path(relative: str) -> str:
    base = Path(__file__).resolve().parent.parent
    return str((base / relative).resolve())


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        persist_dir = _resolve_path(settings.chroma_persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name="medicines",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _medicine_to_document(med: dict) -> str:
    parts = [
        f"Drug Name: {med['name']}",
        f"Generic Name: {med.get('generic_name', 'N/A')}",
        f"Category: {med.get('category', 'N/A')}",
        f"Description: {med.get('description', '')}",
        f"Uses: {med.get('uses', '')}",
        f"Dosage: {med.get('dosage', '')}",
        f"Side Effects: {med.get('side_effects', '')}",
        f"Warnings: {med.get('warnings', '')}",
        f"Interactions: {med.get('interactions', '')}",
    ]
    return "\n".join(parts)


def initialize_vector_store() -> int:
    global _initialized
    collection = get_collection()

    if collection.count() > 0:
        _initialized = True
        return collection.count()

    data_path = _resolve_path(settings.medicines_data_path)
    with open(data_path, encoding="utf-8") as f:
        medicines = json.load(f)

    documents = [_medicine_to_document(m) for m in medicines]
    ids = [f"med_{m['id']}" for m in medicines]
    metadatas = [
        {
            "name": m["name"],
            "category": m.get("category", ""),
            "generic_name": m.get("generic_name", ""),
        }
        for m in medicines
    ]
    embeddings = embed_texts(documents)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    _initialized = True
    return len(medicines)


def search_medicines(query: str, top_k: int | None = None) -> list[dict]:
    collection = get_collection()
    k = top_k or settings.top_k_results

    if collection.count() == 0:
        initialize_vector_store()

    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    items = []
    if not results["documents"] or not results["documents"][0]:
        return items

    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - distance
        if similarity >= settings.similarity_threshold:
            items.append(
                {
                    "document": doc,
                    "metadata": meta,
                    "similarity": round(similarity, 4),
                }
            )

    return items


def build_context(retrieved: list[dict]) -> str:
    if not retrieved:
        return ""
    sections = []
    for i, item in enumerate(retrieved, 1):
        sections.append(
            f"[Source {i}: {item['metadata'].get('name', 'Unknown')} "
            f"(relevance: {item['similarity']:.0%})]\n{item['document']}"
        )
    return "\n\n".join(sections)


def compute_confidence(retrieved: list[dict], has_emergency: bool) -> float:
    if has_emergency:
        return 0.95
    if not retrieved:
        return 0.25
    avg_similarity = sum(r["similarity"] for r in retrieved) / len(retrieved)
    count_factor = min(len(retrieved) / settings.top_k_results, 1.0)
    confidence = (avg_similarity * 0.7) + (count_factor * 0.3)
    return round(min(max(confidence, 0.1), 0.99), 2)
