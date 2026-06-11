"""
scripts/evaluate_retrieval.py
──────────────────────────────
Measures search latency, index sizes, MRR, and Recall@5 over baseline questions.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from app.core import config
from app.services.retrieval_service import search, get_qdrant_client

QUERIES_PATH = Path(__file__).parent.parent / "tests" / "retrieval_queries.json"

def main():
    if not QUERIES_PATH.exists():
        print(f"Error: baseline queries not found at {QUERIES_PATH}")
        sys.exit(1)

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} evaluation queries.")
    
    client = None
    try:
        client = get_qdrant_client()
        try:
            count_res = client.count(collection_name=config.QDRANT_COLLECTION)
            indexed_chunks = count_res.count
        except Exception as e:
            print(f"Error connecting to Qdrant or collection missing: {e}")
            indexed_chunks = 0
            
        print(f"Currently indexed chunks: {indexed_chunks}")

        latencies = []
        hits_at_5 = 0
        reciprocal_ranks = []
        
        # Threshold for a "relevant" chunk (heuristic)
        RELEVANCE_THRESHOLD = 0.50

        for query in queries:
            start_time = time.perf_counter()
            res = search(query=query, top_k=5)
            elapsed = time.perf_counter() - start_time
            
            latencies.append(elapsed * 1000)  # in ms
            
            results = res.get("results", [])
            
            # Calculate heuristics
            # Find first result meeting the relevance threshold
            found_rank = 0
            for r in results:
                if r.get("relevance_score", 0.0) >= RELEVANCE_THRESHOLD:
                    found_rank = r.get("rank", 1)
                    break
                    
            if found_rank > 0:
                hits_at_5 += 1
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)

        # Compute metrics
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        max_latency = max(latencies) if latencies else 0.0
        recall_at_5 = (hits_at_5 / len(queries)) if queries else 0.0
        mrr = (sum(reciprocal_ranks) / len(queries)) if queries else 0.0

        print("\n" + "="*40)
        print("      MEDMITRA RETRIEVAL EVALUATION      ")
        print("="*40)
        print(f"Total Queries Evaluated : {len(queries)}")
        print(f"Total Indexed Chunks    : {indexed_chunks}")
        print(f"Average Latency         : {avg_latency:.2f} ms")
        print(f"Maximum Latency         : {max_latency:.2f} ms")
        print(f"Recall@5 (Score >= {RELEVANCE_THRESHOLD})  : {recall_at_5 * 100:.1f}%")
        print(f"MRR (Mean Reciprocal Rk) : {mrr:.3f}")
        print("="*40)
    finally:
        if client is not None:
            client.close()

if __name__ == "__main__":
    main()
