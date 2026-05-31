"""
scripts/build_medicine_index.py
─────────────────────────────────
Builds the ChromaDB vector index for medicines.

Run AFTER create_prototype_data.py.

Usage:
    python scripts\\build_medicine_index.py           # skip if index exists
    python scripts\\build_medicine_index.py --reset   # wipe and rebuild
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT    = Path(__file__).parent.parent
CHROMA_PATH     = PROJECT_ROOT / "chroma_db"
PROTOTYPE_PATH  = PROJECT_ROOT / "data" / "prototype_medicine_data.csv"
COLLECTION_NAME = "medmitra_medicines"
MODEL_NAME      = "NeuML/pubmedbert-base-embeddings"
BATCH_SIZE      = 64

SEP = "=" * 65


def parse_args():
    p = argparse.ArgumentParser(description="Build the medicine vector index.")
    p.add_argument("--reset", action="store_true",
                   help="Delete existing collection and rebuild from scratch.")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\n{SEP}")
    print("  MedMitra — Build Medicine Index")
    print(SEP)

    if not PROTOTYPE_PATH.exists():
        print(f"\n❌ Prototype file not found: {PROTOTYPE_PATH}")
        print("   Run: python scripts\\create_prototype_data.py")
        sys.exit(1)

    df = pd.read_csv(PROTOTYPE_PATH)
    print(f"\n📂 Prototype file: {PROTOTYPE_PATH}")
    print(f"   Rows      : {len(df):,}")
    print(f"   Medicines : {df['medicine'].nunique():,}")

    empty = df["content"].str.strip().eq("").sum()
    if empty > 0:
        print(f"⚠️  {empty} rows have empty content — dropping them")
        df = df[df["content"].str.strip() != ""]

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

    try:
        existing = client.get_collection(name=COLLECTION_NAME)
        count = existing.count()
        if count > 0 and not args.reset:
            print(f"\n✅ Collection '{COLLECTION_NAME}' already exists with {count:,} records.")
            print("   Use --reset to rebuild. Exiting.")
            return
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"\n🔍 Loading model: {MODEL_NAME}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    print("   Model loaded ✅")

    documents = df["content"].tolist()
    ids       = df["id"].tolist()
    metadatas = df.apply(lambda r: {
        "title":          r["medicine"],
        "category":       r.get("category",       ""),
        "indication":     r.get("indication",     ""),
        "dosage_form":    r.get("dosage_form",    ""),
        "classification": r.get("classification", ""),
        "source_type":    "medicine",
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

        embeddings = model.encode(
            batch_docs,
            normalize_embeddings=True,
            show_progress_bar=False,
            
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
    print("  DONE. Next step: python scripts\\test_medicine_search.py")
    print(SEP)


if __name__ == "__main__":
    main()
