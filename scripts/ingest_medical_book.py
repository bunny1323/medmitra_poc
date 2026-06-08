"""
scripts/ingest_medical_book.py
──────────────────────────────
Reads the medical book PDF, chunks the text, computes Dense and Sparse embeddings,
and stores them in a Qdrant collection for Hybrid Search.
"""

import os
import sys
import argparse
from pathlib import Path

# Fix path to allow importing app modules
sys.path.append(str(Path(__file__).parent.parent))

import time
from dotenv import load_dotenv
load_dotenv()

import fitz  # PyMuPDF
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Using fastembed for BOTH Dense and Sparse to avoid scikit-learn/PyTorch DLL blocks on Windows
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# Configuration
PDF_PATH = Path(__file__).parent.parent / "data" / "current-medical-diagnosis-and-treatment-2025-1.pdf"
COLLECTION_NAME = "medical_book_2025"
QDRANT_PATH = Path(__file__).parent.parent / "qdrant_db"

DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def parse_pdf() -> list[dict]:
    """Parse PDF and return text with metadata."""
    if not PDF_PATH.exists():
        print(f"Error: PDF not found at {PDF_PATH}")
        sys.exit(1)

    print(f"Parsing PDF: {PDF_PATH.name}")
    doc = fitz.open(PDF_PATH)
    pages_data = []

    for page_num in tqdm(range(len(doc)), desc="Reading pages"):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages_data.append({
                "text": text,
                "metadata": {
                    "source": PDF_PATH.name,
                    "page": page_num + 1,
                }
            })
    return pages_data

def chunk_text(pages_data: list[dict]) -> list[dict]:
    """Chunk the extracted text using Langchain."""
    print("Chunking text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = []
    chunk_id = 1
    for page_data in tqdm(pages_data, desc="Chunking pages"):
        texts = splitter.split_text(page_data["text"])
        for text in texts:
            chunks.append({
                "id": chunk_id,
                "content": text,
                "metadata": page_data["metadata"].copy()
            })
            chunk_id += 1
            
    print(f"Created {len(chunks)} chunks.")
    return chunks

def build_index(chunks: list[dict]):
    """Embed chunks and store in Qdrant."""
    print("Initializing embedding models (using fastembed to bypass DLL policies)...")
    dense_model = TextEmbedding(DENSE_MODEL_NAME)
    sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url and qdrant_api_key:
        print("Using Qdrant Cloud...")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=300.0)
    else:
        print("Using local Qdrant...")
        client = QdrantClient(path=str(QDRANT_PATH), timeout=300.0)
    
    # Check if collection exists
    collections = client.get_collections().collections
    if any(c.name == COLLECTION_NAME for c in collections):
        print(f"Recreating collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
    else:
        print(f"Creating collection '{COLLECTION_NAME}'...")

    # Fastembed BAAI/bge-small-en-v1.5 has size 384
    dense_vector_size = 384 
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": qmodels.VectorParams(
                size=dense_vector_size,
                distance=qmodels.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": qmodels.SparseVectorParams(
                modifier=qmodels.Modifier.IDF
            )
        }
    )

    BATCH_SIZE = 15
    print(f"Embedding and uploading {len(chunks)} chunks in batches of {BATCH_SIZE}...")
    
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Uploading to Qdrant"):
        batch = chunks[i:i + BATCH_SIZE]
        contents = [item["content"] for item in batch]
        ids = [item["id"] for item in batch]
        
        # Dense embeddings
        dense_gen = dense_model.embed(contents)
        dense_embeddings = [vec.tolist() for vec in dense_gen]
        
        # Sparse embeddings
        sparse_gen = sparse_model.embed(contents)
        sparse_embeddings = []
        for sparse_vec in sparse_gen:
            sparse_embeddings.append(
                qmodels.SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist()
                )
            )
            
        points = []
        for j in range(len(batch)):
            payload = batch[j]["metadata"]
            payload["content"] = batch[j]["content"]
            
            points.append(qmodels.PointStruct(
                id=ids[j],
                vector={
                    "dense": dense_embeddings[j],
                    "sparse": sparse_embeddings[j]
                },
                payload=payload
            ))
            
        max_retries = 5
        for attempt in range(max_retries):
            try:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                break
            except Exception as e:
                print(f"\\nUpsert failed: {e}. Retrying {attempt+1}/{max_retries} in 5s...")
                time.sleep(5.0)
        else:
            print("\\nFailed to upload batch after max retries.")
            
        time.sleep(1.0)  # Pause to avoid rate limits and connection drops

    print("✅ Ingestion complete.")

def main():
    parser = argparse.ArgumentParser(description="Ingest Medical Book into Qdrant")
    args = parser.parse_args()
    
    pages = parse_pdf()
    chunks = chunk_text(pages)
    build_index(chunks)

if __name__ == "__main__":
    main()
