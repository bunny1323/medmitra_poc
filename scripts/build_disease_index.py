"""
scripts/build_disease_index.py
────────────────────────────────
Builds the ChromaDB vector index for diseases.

Run AFTER create_prototype_data.py.

This script ONLY runs when you explicitly call it — never from Streamlit.
Embeddings are stored persistently in chroma_db/ and reused at query time.

Usage:
    python scripts\\build_disease_index.py           # skip if index exists
    python scripts\\build_disease_index.py --reset   # wipe and rebuild

⚠️  First run downloads NeuML/pubmedbert-base-embeddings (~440 MB).
   Subsequent runs use the local HuggingFace cache.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# ── Make imports work from the project root ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT    = Path(__file__).parent.parent
CHROMA_PATH     = PROJECT_ROOT / "chroma_db"
PROTOTYPE_PATH  = PROJECT_ROOT / "data" / "prototype_disease_data.csv"
COLLECTION_NAME = "medmitra_diseases"
MODEL_NAME      = "NeuML/pubmedbert-base-embeddings"
BATCH_SIZE      = 64

SEP = "=" * 65


def parse_args():
    p = argparse.ArgumentParser(description="Build the disease vector index.")
    p.add_argument("--reset", action="store_true",
                   help="Delete existing collection and rebuild from scratch.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{SEP}")
    print("  MedMitra — Build Disease Index")
    print(SEP)

    # ── 1. Validate input file ─────────────────────────────────────────────────
    if not PROTOTYPE_PATH.exists():
        print(f"\n❌ Prototype file not found: {PROTOTYPE_PATH}")
        print("   Run: python scripts\\create_prototype_data.py")
        sys.exit(1)

    df = pd.read_csv(PROTOTYPE_PATH)
    print(f"\n📂 Prototype file: {PROTOTYPE_PATH}")
    print(f"   Rows   : {len(df):,}")
    print(f"   Diseases: {df['disease'].nunique():,}")

    # Validate no empty content
    empty = df["content"].str.strip().eq("").sum()
    if empty > 0:
        print(f"⚠️  {empty} rows have empty content — dropping them")
        df = df[df["content"].str.strip() != ""]

    # ── 2. Open ChromaDB ──────────────────────────────────────────────────────
    import chromadb
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if args.reset:
        print(f"\n🔄 --reset: deleting existing collection '{COLLECTION_NAME}' …")
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print("   Deleted.")
        except Exception:
            print("   Collection did not exist — skipping delete.")

    # Check if already built
    try:
        existing = client.get_collection(name=COLLECTION_NAME)
        count = existing.count()
        if count > 0 and not args.reset:
            print(f"\n✅ Collection '{COLLECTION_NAME}' already exists with {count:,} records.")
            print("   Use --reset to rebuild. Exiting.")
            return
    except Exception:
        pass  # Does not exist yet — continue to build

    # Create collection with cosine similarity
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ── 3. Load embedding model ────────────────────────────────────────────────
    print(f"\n🔍 Loading model: {MODEL_NAME}")
    print("   (First run downloads ~440 MB — subsequent runs use cache)")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print("   Model loaded ✅")

    # ── 4. Embed and index in batches ─────────────────────────────────────────
    documents = df["content"].tolist()
    ids       = df["id"].tolist()
    metadatas = df.apply(lambda r: {
        "title":          r["disease"],
        "symptoms":       r.get("symptoms", ""),
        "source_type":    "disease",
        "source_dataset": r.get("source_dataset", "unknown"),
    }, axis=1).tolist()

    total   = len(documents)
    added   = 0
    t_start = time.time()

    print(f"\n⚙️  Indexing {total:,} documents in batches of {BATCH_SIZE} …\n")

    for start in range(0, total, BATCH_SIZE):
        end        = min(start + BATCH_SIZE, total)
        batch_docs = documents[start:end]
        batch_ids  = ids[start:end]
        batch_meta = metadatas[start:end]

        # Embed with L2 normalization (important for cosine similarity)

        embeddings = model.encode(
            batch_docs,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()

        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_meta,
        )
        added += len(batch_docs)
        elapsed = time.time() - t_start
        pct = 100 * added / total
        rate = added / elapsed if elapsed > 0 else 0
        print(f"  [{added:>5}/{total}]  {pct:5.1f}%  |  {rate:.0f} docs/sec", end="\r")

    elapsed = time.time() - t_start
    print(f"\n\n✅ Indexed {added:,} documents in {elapsed:.1f}s")
    print(f"   Collection '{COLLECTION_NAME}' now has {collection.count():,} records")
    print(f"   Stored in: {CHROMA_PATH}")

    print(f"\n{SEP}")
    print("  DONE. Next step: python scripts\\test_disease_search.py")
    print(SEP)


if __name__ == "__main__":
    main()
