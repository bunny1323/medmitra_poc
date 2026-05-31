"""
scripts/test_disease_search.py
────────────────────────────────
Tests disease retrieval WITHOUT calling Llama.

Run AFTER build_disease_index.py.
Confirms that semantic search returns sensible results.

Usage:
    python scripts\\test_disease_search.py

IMPORTANT: Relevance scores are cosine similarity scores — they measure
how semantically similar the query is to indexed records.
They are NOT diagnosis probabilities.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.retrieval_service import search, collection_exists, DISEASE_COLLECTION

SEP = "=" * 65

TEST_QUERIES = [
    "I have fever, cough, body pain and weakness",
    "I have runny nose, sneezing and sore throat",
    "I have severe headache and sensitivity to light",
    "I have wheezing and chest tightness",
    "I have stomach pain and vomiting",
    "feeling anxious, palpitations and shortness of breath",
    "joint pain and swelling in fingers",
]


def run_tests():
    print(f"\n{SEP}")
    print("  MedMitra — Disease Retrieval Test (no Llama)")
    print(SEP)

    if not collection_exists(DISEASE_COLLECTION):
        print(f"\n❌ Collection '{DISEASE_COLLECTION}' not found or empty.")
        print("   Run: python scripts\\build_disease_index.py --reset")
        sys.exit(1)

    print(f"\n✅ Collection '{DISEASE_COLLECTION}' found.\n")
    print("⚠️  REMINDER: Relevance scores are semantic similarity, NOT diagnosis probability.\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─'*65}")
        print(f"  Query {i}: {query}")
        print("─"*65)

        result = search(query=query, collection_name=DISEASE_COLLECTION, top_k=3)

        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
            continue

        print(f"  Top relevance: {result['top_relevance']:.3f}  (threshold for Llama: 0.55)")
        print()

        for r in result["results"]:
            print(f"  Rank {r['rank']} | Relevance: {r['relevance_score']:.3f} | Distance: {r['distance']:.4f}")
            print(f"  Disease : {r['metadata'].get('title', 'unknown')}")
            print(f"  Content : {r['content'][:120]}…")
            print()

    print(f"\n{'─'*65}")
    print(f"  {result['warning']}")
    print(f"{'─'*65}")
    print(f"\n{SEP}")
    print("  TEST COMPLETE.")
    print("  If results look relevant → proceed to build_medicine_index.py")
    print("  If results look wrong → check clean_disease_data.py output")
    print(SEP)


if __name__ == "__main__":
    run_tests()
