"""
scripts/test_medicine_search.py
─────────────────────────────────
Tests medicine retrieval WITHOUT calling Llama.

Run AFTER build_medicine_index.py.

NOTE: The medicine dataset is SYNTHETIC. Real drug names like 'paracetamol'
will return results based on category/indication similarity, not exact name
match, because the dataset uses invented names. This is expected behaviour
for the prototype.

Usage:
    python scripts\\test_medicine_search.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.retrieval_service import search, collection_exists, MEDICINE_COLLECTION

SEP = "=" * 65

TEST_QUERIES = [
    "medicine for fever and pain relief",
    "antibiotic for infection",
    "antifungal treatment",
    "medicine for depression",
    "antiviral for virus infection",
    "medicine for diabetes",
    "antiseptic wound care",
    "cetirizine allergy medicine",    # real name — will match by category
    "paracetamol fever relief",       # real name — will match by indication
]


def run_tests():
    print(f"\n{SEP}")
    print("  MedMitra — Medicine Retrieval Test (no Llama)")
    print(SEP)

    if not collection_exists(MEDICINE_COLLECTION):
        print(f"\n❌ Collection '{MEDICINE_COLLECTION}' not found or empty.")
        print("   Run: python scripts\\build_medicine_index.py --reset")
        sys.exit(1)

    print(f"\n✅ Collection '{MEDICINE_COLLECTION}' found.")
    print(f"\n⚠️  NOTE: Medicine dataset is SYNTHETIC — names are invented.")
    print("   Retrieval is by category/indication similarity, not exact drug name.\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─'*65}")
        print(f"  Query {i}: {query}")
        print("─"*65)

        result = search(query=query, collection_name=MEDICINE_COLLECTION, top_k=3)

        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
            continue

        print(f"  Top relevance: {result['top_relevance']:.3f}  (threshold for Llama: 0.55)")
        print()

        for r in result["results"]:
            meta = r["metadata"]
            print(f"  Rank {r['rank']} | Relevance: {r['relevance_score']:.3f} | Distance: {r['distance']:.4f}")
            print(f"  Medicine    : {meta.get('title','?')}  |  Category: {meta.get('category','?')}")
            print(f"  Indication  : {meta.get('indication','?')}  |  Form: {meta.get('dosage_form','?')}")
            print(f"  Content     : {r['content'][:100]}…")
            print()

    print(f"\n{'─'*65}")
    print("  ⚕️  Relevance scores indicate semantic similarity only.")
    print("   They are NOT recommendations and do NOT confirm any medical decision.")
    print(f"{'─'*65}")
    print(f"\n{SEP}")
    print("  TEST COMPLETE.")
    print("  Next step: add GROQ_API_KEY to .env → python -m streamlit run app.py")
    print(SEP)


if __name__ == "__main__":
    run_tests()
