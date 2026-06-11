from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models as qmodels

from app.core import config
from app.services.retrieval_service import (
    get_dense_model,
    get_qdrant_client,
    get_sparse_model,
)


# ---------------------------------------------------------------------
# Local storage paths
# ---------------------------------------------------------------------

BOOKS_DIR = config.BASE_DIR / "data" / "books"
REGISTRY_DIR = config.BASE_DIR / "data" / "registry"
REGISTRY_FILE = REGISTRY_DIR / "documents.json"


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def to_python_list(value: Any) -> list:
    """
    Convert NumPy arrays, FastEmbed values, or normal Python iterables
    into plain Python lists.
    """
    if hasattr(value, "tolist"):
        return value.tolist()

    return list(value)


def calculate_sha256(file_path: Path) -> str:
    """
    Generate a SHA-256 fingerprint for duplicate detection.
    """
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(8192), b""):
            sha256.update(block)

    return sha256.hexdigest()


def make_display_name(filename: str) -> str:
    """
    Convert a PDF filename into a readable source name.
    """
    name_without_extension = os.path.splitext(filename)[0]

    spaced_name = (
        name_without_extension
        .replace("_", " ")
        .replace("-", " ")
    )

    special_terms = {
        "icmr",
        "stw",
        "who",
        "imci",
        "pdf",
        "rag",
        "i",
        "ii",
        "iii",
    }

    words = []

    for word in spaced_name.split():
        if word.lower() in special_terms:
            words.append(word.upper())
        else:
            words.append(word.capitalize())

    return " ".join(words)


def clean_text(text: str) -> str:
    """
    Normalize extracted page text before chunking.
    """
    return re.sub(r"\s+", " ", text).strip()


def make_point_id(
    source_id: str,
    chunk_index: int,
) -> str:
    """
    Generate a deterministic Qdrant point ID.

    Re-running ingestion for the same book and chunk index will produce
    the same point ID, making retries safe and idempotent.
    """
    return str(
        uuid.uuid5(
            uuid.UUID(source_id),
            f"chunk_{chunk_index}",
        )
    )


# ---------------------------------------------------------------------
# Registry management
# ---------------------------------------------------------------------

def load_registry() -> list[dict[str, Any]]:
    """
    Load book-ingestion metadata from the local JSON registry.
    """
    if not REGISTRY_FILE.exists():
        REGISTRY_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_registry([])

        return []

    try:
        with open(
            REGISTRY_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            registry = json.load(file)

        return registry if isinstance(registry, list) else []

    except Exception:
        return []


def save_registry(
    registry: list[dict[str, Any]],
) -> None:
    """
    Persist book-ingestion metadata.
    """
    REGISTRY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REGISTRY_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            registry,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------
# Qdrant collection helpers
# ---------------------------------------------------------------------

def collection_exists() -> bool:
    """
    Return True when the configured Qdrant collection exists.
    """
    client = get_qdrant_client()

    collections = client.get_collections().collections

    return any(
        collection.name == config.QDRANT_COLLECTION
        for collection in collections
    )


def initialize_collection_if_needed() -> None:
    """
    Create the Qdrant collection when required.

    HNSW indexing is temporarily disabled using m=0 while bulk-uploading
    points. This reduces CPU and memory pressure on a small cloud cluster.
    """
    client = get_qdrant_client()

    if collection_exists():
        return

    print(
        f"[ingestion] Creating Qdrant collection "
        f"'{config.QDRANT_COLLECTION}'..."
    )

    client.create_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=config.DENSE_VECTOR_SIZE,
                distance=qmodels.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams()
        },
        hnsw_config=qmodels.HnswConfigDiff(
            m=0,
        ),
    )

    try:
        client.create_payload_index(
            collection_name=config.QDRANT_COLLECTION,
            field_name="source_id",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    except Exception as exc:
        print(
            "[ingestion] Warning: could not create payload index "
            f"for source_id: {exc}"
        )


def enable_hnsw_indexing() -> None:
    """
    Enable normal HNSW graph construction after bulk ingestion.
    """
    if not collection_exists():
        return

    client = get_qdrant_client()

    print("[ingestion] Enabling HNSW indexing...")

    client.update_collection(
        collection_name=config.QDRANT_COLLECTION,
        hnsw_config=qmodels.HnswConfigDiff(
            m=config.HNSW_M,
        ),
    )


def delete_book_from_qdrant(
    source_id: str,
) -> None:
    """
    Delete only the chunks belonging to one PDF.

    Other books and their vectors remain unchanged.
    """
    if not collection_exists():
        return

    client = get_qdrant_client()

    client.delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_id",
                        match=qmodels.MatchValue(
                            value=source_id,
                        ),
                    )
                ]
            )
        ),
        wait=True,
    )


def upload_batch_with_retry(
    points: list[qmodels.PointStruct],
    batch_number: int,
    total_batches: int,
    processed_chunks: int,
    total_chunks: int,
) -> None:
    """
    Upload one batch with conservative cloud-friendly settings.
    """
    client = get_qdrant_client()

    print(
        f"[ingestion] Uploading batch "
        f"{batch_number}/{total_batches} "
        f"| chunks {processed_chunks}/{total_chunks}"
    )

    for attempt in range(
        1,
        config.INGEST_MAX_RETRIES + 1,
    ):
        try:
            client.upload_points(
                collection_name=config.QDRANT_COLLECTION,
                points=points,
                batch_size=len(points),
                parallel=1,
                max_retries=1,
                wait=True,
            )

            return

        except Exception as exc:
            if attempt >= config.INGEST_MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to upload batch "
                    f"{batch_number}/{total_batches} "
                    f"after {config.INGEST_MAX_RETRIES} attempts: {exc}"
                ) from exc

            delay_seconds = min(
                config.INGEST_RETRY_BASE_SECONDS * attempt,
                30,
            )

            print(
                f"[ingestion] Upload attempt "
                f"{attempt}/{config.INGEST_MAX_RETRIES} failed: {exc}"
            )

            print(
                f"[ingestion] Retrying after "
                f"{delay_seconds:.1f} seconds..."
            )

            time.sleep(delay_seconds)


# ---------------------------------------------------------------------
# PDF processing and indexing
# ---------------------------------------------------------------------

def extract_chunks_from_pdf(
    file_path: Path,
) -> list[dict[str, Any]]:
    """
    Extract page-level text and split it into overlapping chunks.
    """
    chunks: list[dict[str, Any]] = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunk_index = 0

    with fitz.open(file_path) as document:
        for page_number in range(len(document)):
            page = document[page_number]

            page_text = clean_text(
                page.get_text()
            )

            if not page_text:
                continue

            page_chunks = splitter.split_text(
                page_text
            )

            for chunk_text in page_chunks:
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "page": page_number + 1,
                        "content": chunk_text,
                    }
                )

                chunk_index += 1

    return chunks


def ingest_pdf_file(
    file_path: Path,
    source_id: str,
    display_name: str,
) -> int:
    """
    Process one PDF and upload its dense and sparse vectors to Qdrant.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    print(
        f"[ingestion] Processing: {file_path.name}"
    )

    chunks = extract_chunks_from_pdf(
        file_path
    )

    if not chunks:
        print(
            f"[ingestion] No text found in "
            f"{file_path.name}. Skipping."
        )

        return 0

    initialize_collection_if_needed()

    dense_model = get_dense_model()
    sparse_model = get_sparse_model()

    batch_size = config.INGEST_BATCH_SIZE
    total_chunks = len(chunks)

    total_batches = (
        total_chunks + batch_size - 1
    ) // batch_size

    print(
        f"[ingestion] Indexing {total_chunks} chunks "
        f"in {total_batches} batches of {batch_size}..."
    )

    for batch_start in range(
        0,
        total_chunks,
        batch_size,
    ):
        batch = chunks[
            batch_start:
            batch_start + batch_size
        ]

        contents = [
            chunk["content"]
            for chunk in batch
        ]

        dense_embeddings = [
            to_python_list(vector)
            for vector in dense_model.embed(contents)
        ]

        sparse_embeddings = []

        for vector in sparse_model.embed(contents):
            sparse_embeddings.append(
                qmodels.SparseVector(
                    indices=to_python_list(
                        vector.indices
                    ),
                    values=to_python_list(
                        vector.values
                    ),
                )
            )

        points: list[qmodels.PointStruct] = []

        for index, chunk in enumerate(batch):
            point_id = make_point_id(
                source_id=source_id,
                chunk_index=chunk["chunk_index"],
            )

            payload = {
                "source_id": source_id,
                "source_name": display_name,
                "original_filename": file_path.name,
                "source_type": "official_guideline",
                "page": chunk["page"],
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
            }

            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_embeddings[index],
                        "sparse": sparse_embeddings[index],
                    },
                    payload=payload,
                )
            )

        batch_number = (
            batch_start // batch_size
        ) + 1

        processed_chunks = min(
            batch_start + len(batch),
            total_chunks,
        )

        upload_batch_with_retry(
            points=points,
            batch_number=batch_number,
            total_batches=total_batches,
            processed_chunks=processed_chunks,
            total_chunks=total_chunks,
        )

    return total_chunks


# ---------------------------------------------------------------------
# Public ingestion operations
# ---------------------------------------------------------------------

def append_books() -> dict[str, Any]:
    """
    Scan data/books and ingest only new or changed PDFs.

    Behaviour:
        new PDF        -> append only that PDF
        same PDF hash  -> skip
        same filename with new hash -> replace only that PDF
    """
    BOOKS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_files = sorted(
        BOOKS_DIR.glob("*.pdf")
    )

    registry = load_registry()

    ingested_count = 0
    replaced_count = 0
    skipped_count = 0

    results: list[dict[str, Any]] = []

    for file_path in pdf_files:
        file_hash = calculate_sha256(
            file_path
        )

        existing_by_hash = next(
            (
                item
                for item in registry
                if item.get("document_hash") == file_hash
            ),
            None,
        )

        existing_by_name = next(
            (
                item
                for item in registry
                if item.get("original_filename") == file_path.name
            ),
            None,
        )

        if (
            existing_by_hash
            and existing_by_hash.get("ingestion_status") == "ingested"
        ):
            print(
                f"[ingestion] Skipping duplicate PDF: "
                f"{file_path.name}"
            )

            skipped_count += 1

            results.append(
                {
                    "filename": file_path.name,
                    "status": "skipped",
                    "reason": "Duplicate SHA-256 hash",
                }
            )

            continue

        if existing_by_name:
            print(
                f"[ingestion] Updated PDF detected: "
                f"{file_path.name}"
            )

            replace_book(
                source_id=existing_by_name["source_id"],
                finalize_indexing=False,
            )

            replaced_count += 1

            results.append(
                {
                    "filename": file_path.name,
                    "status": "replaced",
                }
            )

            continue

        source_id = str(
            uuid.uuid4()
        )

        display_name = make_display_name(
            file_path.name
        )

        with fitz.open(file_path) as document:
            page_count = len(document)

        registry_entry = {
            "source_id": source_id,
            "display_name": display_name,
            "original_filename": file_path.name,
            "document_hash": file_hash,
            "ingestion_status": "pending",
            "page_count": page_count,
            "chunk_count": 0,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "error_message": None,
        }

        registry.append(
            registry_entry
        )

        save_registry(
            registry
        )

        try:
            chunk_count = ingest_pdf_file(
                file_path=file_path,
                source_id=source_id,
                display_name=display_name,
            )

            registry_entry["ingestion_status"] = "ingested"
            registry_entry["chunk_count"] = chunk_count
            registry_entry["updated_at"] = utc_now_iso()
            registry_entry["error_message"] = None

            save_registry(
                registry
            )

            ingested_count += 1

            results.append(
                {
                    "filename": file_path.name,
                    "status": "ingested",
                    "chunk_count": chunk_count,
                }
            )

        except Exception as exc:
            registry_entry["ingestion_status"] = "failed"
            registry_entry["updated_at"] = utc_now_iso()
            registry_entry["error_message"] = str(exc)

            save_registry(
                registry
            )

            print(
                f"[ingestion] Failed to ingest "
                f"{file_path.name}: {exc}"
            )

            results.append(
                {
                    "filename": file_path.name,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    if (
        ingested_count > 0
        or replaced_count > 0
    ):
        enable_hnsw_indexing()

    return {
        "mode": "append",
        "ingested_count": ingested_count,
        "replaced_count": replaced_count,
        "skipped_count": skipped_count,
        "results": results,
    }


def replace_book(
    source_id: str,
    finalize_indexing: bool = True,
) -> dict[str, Any]:
    """
    Delete one PDF's existing vectors and index its updated version.
    """
    registry = load_registry()

    entry = next(
        (
            item
            for item in registry
            if item.get("source_id") == source_id
        ),
        None,
    )

    if not entry:
        raise ValueError(
            f"No registered book found with source_id: {source_id}"
        )

    file_path = (
        BOOKS_DIR
        / entry["original_filename"]
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"PDF source file not found: {file_path}"
        )

    print(
        f"[ingestion] Replacing book: "
        f"{entry['display_name']}"
    )

    delete_book_from_qdrant(
        source_id
    )

    entry["ingestion_status"] = "pending"
    entry["document_hash"] = calculate_sha256(
        file_path
    )
    entry["updated_at"] = utc_now_iso()
    entry["error_message"] = None

    save_registry(
        registry
    )

    try:
        chunk_count = ingest_pdf_file(
            file_path=file_path,
            source_id=source_id,
            display_name=entry["display_name"],
        )

        entry["ingestion_status"] = "ingested"
        entry["chunk_count"] = chunk_count
        entry["updated_at"] = utc_now_iso()
        entry["error_message"] = None

        save_registry(
            registry
        )

        if finalize_indexing:
            enable_hnsw_indexing()

        return {
            "status": "success",
            "message": (
                f"Replaced book "
                f"'{entry['display_name']}' successfully."
            ),
            "chunk_count": chunk_count,
        }

    except Exception as exc:
        entry["ingestion_status"] = "failed"
        entry["updated_at"] = utc_now_iso()
        entry["error_message"] = str(exc)

        save_registry(
            registry
        )

        raise


def delete_book(
    source_id: str,
) -> dict[str, Any]:
    """
    Remove one PDF and only its vectors from Qdrant.
    """
    registry = load_registry()

    entry = next(
        (
            item
            for item in registry
            if item.get("source_id") == source_id
        ),
        None,
    )

    if not entry:
        raise ValueError(
            f"No registered book found with source_id: {source_id}"
        )

    print(
        f"[ingestion] Deleting book: "
        f"{entry['display_name']}"
    )

    delete_book_from_qdrant(
        source_id
    )

    updated_registry = [
        item
        for item in registry
        if item.get("source_id") != source_id
    ]

    save_registry(
        updated_registry
    )

    return {
        "status": "success",
        "message": (
            f"Deleted book "
            f"'{entry['display_name']}' successfully."
        ),
    }


def full_rebuild() -> dict[str, Any]:
    """
    Delete the complete collection and ingest every PDF again.
    """
    client = get_qdrant_client()

    print(
        f"[ingestion] Dropping collection "
        f"'{config.QDRANT_COLLECTION}'..."
    )

    if collection_exists():
        client.delete_collection(
            collection_name=config.QDRANT_COLLECTION
        )

    save_registry([])

    result = append_books()

    result["mode"] = "rebuild"

    return result