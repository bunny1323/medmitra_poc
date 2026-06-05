#!/usr/bin/env python3
"""
MedMitra Qdrant Index Builder
================================
Reads processed disease and medicine JSON files, embeds them with MedCPT
Article Encoder (dense) and fastembed BM25 (sparse), and upserts all records
into the Qdrant 'medmitra_knowledge' collection.

Run AFTER clean_datasets.py and AFTER starting Qdrant:
    docker run -p 6333:6333 qdrant/qdrant

Usage:
    python -m app.scripts.build_qdrant_index

Options (via env vars):
    QDRANT_URL          Qdrant server URL (default: http://localhost:6333)
    QDRANT_API_KEY      Optional cloud API key
    QDRANT_COLLECTION   Collection name (default: medmitra_knowledge)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("build_qdrant_index")

# Ensure app is importable when run as module or script
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))


def load_records(path: str, label: str) -> List[Dict]:
    if not os.path.exists(path):
        log.warning(f"{label} file not found: {path}. Run clean_datasets.py first.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    log.info(f"Loaded {len(records)} {label} records from {path}")
    return records


def embed_dense(records: List[Dict], medcpt_retriever) -> List[List[float]]:
    """Embed search_text of each record using MedCPT Article Encoder."""
    texts = [r.get("search_text") or r.get("description") or r.get("medicine_name", "") for r in records]

    log.info(f"Embedding {len(texts)} records with MedCPT Article Encoder...")
    # MedCPT Article Encoder expects list of strings or [title, text] pairs
    embeddings = medcpt_retriever.embed_articles(texts)
    log.info(f"Dense embeddings shape: {embeddings.shape}")
    return [emb.tolist() for emb in embeddings]


def embed_sparse(records: List[Dict], sparse_bm25_service) -> List[Dict]:
    """Embed search_text of each record using fastembed BM25."""
    texts = [r.get("search_text") or r.get("description") or r.get("medicine_name", "") for r in records]
    log.info(f"Generating BM25 sparse vectors for {len(texts)} records...")
    sparse_vecs = sparse_bm25_service.embed_texts(texts)
    non_empty = sum(1 for v in sparse_vecs if v["indices"])
    log.info(f"Sparse vectors generated: {non_empty}/{len(sparse_vecs)} non-empty")
    return sparse_vecs


def main():
    from app.core.config import settings

    log.info("=" * 60)
    log.info("MedMitra Qdrant Index Builder — Starting")
    log.info("=" * 60)

    # Load processed records
    disease_records = load_records(settings.processed_diseases_path, "disease")
    medicine_records = load_records(settings.processed_medicines_path, "medicine")

    if not disease_records and not medicine_records:
        log.error(
            "No records to index. Run the data cleaning pipeline first:\n"
            "  python -m app.scripts.clean_datasets\n"
            "Then download Kaggle datasets into app/data/raw/"
        )
        sys.exit(1)

    all_records = disease_records + medicine_records
    log.info(f"Total records to index: {len(all_records)} ({len(disease_records)} diseases, {len(medicine_records)} medicines)")

    # Initialize services
    from app.services.medcpt_retriever import MedCPTRetriever
    from app.services.sparse_bm25_service import SparseBM25Service
    from app.services.qdrant_service import QdrantService

    qdrant_svc = QdrantService()
    if not qdrant_svc.is_connected():
        log.error(
            f"Cannot connect to Qdrant at {settings.QDRANT_URL}.\n"
            "Start Qdrant first:\n"
            "  # Linux/Mac:\n"
            "  docker run -p 6333:6333 qdrant/qdrant\n"
            "  # Windows PowerShell:\n"
            "  docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant"
        )
        sys.exit(1)

    medcpt = MedCPTRetriever()
    sparse_svc = SparseBM25Service()

    # Create collection
    collection = settings.QDRANT_COLLECTION
    log.info(f"Creating collection '{collection}' (will recreate if exists)...")
    qdrant_svc.create_or_recreate_collection(collection, vector_size=settings.DENSE_VECTOR_SIZE)
    log.info(f"Collection '{collection}' ready.")

    # Embed — dense
    log.info("Step 1/3: Generating dense MedCPT embeddings...")
    dense_vecs = embed_dense(all_records, medcpt)

    # Embed — sparse
    log.info("Step 2/3: Generating BM25 sparse vectors...")
    sparse_vecs = embed_sparse(all_records, sparse_svc)

    # Upsert
    log.info("Step 3/3: Upserting records into Qdrant...")
    qdrant_svc.upsert_records(
        collection_name=collection,
        records=all_records,
        dense_vectors=dense_vecs,
        sparse_vectors=sparse_vecs,
        batch_size=64
    )

    # Verify counts
    total = qdrant_svc.get_collection_points_count(collection)
    d_count = qdrant_svc.get_record_type_count(collection, "disease")
    m_count = qdrant_svc.get_record_type_count(collection, "medicine")

    log.info("=" * 60)
    log.info("Qdrant Index Build Complete")
    log.info(f"  Collection      : {collection}")
    log.info(f"  Total points    : {total}")
    log.info(f"  Disease records : {d_count}")
    log.info(f"  Medicine records: {m_count}")
    log.info(f"  Dense vectors   : ✓ (MedCPT Article Encoder)")
    log.info(f"  Sparse vectors  : ✓ (fastembed BM25)")
    log.info("  review_status   : prototype_unverified")
    log.info("  ⚠ Kaggle data is NOT clinically verified.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
