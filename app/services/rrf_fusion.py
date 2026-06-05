from typing import List, Dict, Any

class RRFFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, dense_results: List[Dict[str, Any]], bm25_results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        # Map to hold chunk details by chunk_id
        fused = {}

        # Process dense results
        for item in dense_results:
            cid = item["chunk_id"]
            if cid not in fused:
                fused[cid] = {
                    "chunk_id": cid,
                    "text": item["text"],
                    "source_name": item["source_name"],
                    "source_type": item["source_type"],
                    "page_number": item["page_number"],
                    "dense_rank": item["dense_rank"],
                    "dense_score": item["dense_score"],
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_score": 0.0
                }
            fused[cid]["rrf_score"] += 1.0 / (self.k + item["dense_rank"])

        # Process BM25 results
        for item in bm25_results:
            cid = item["chunk_id"]
            if cid not in fused:
                fused[cid] = {
                    "chunk_id": cid,
                    "text": item["text"],
                    "source_name": item["source_name"],
                    "source_type": item["source_type"],
                    "page_number": item["page_number"],
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": item["bm25_rank"],
                    "bm25_score": item["bm25_score"],
                    "rrf_score": 0.0
                }
            else:
                fused[cid]["bm25_rank"] = item["bm25_rank"]
                fused[cid]["bm25_score"] = item["bm25_score"]
            fused[cid]["rrf_score"] += 1.0 / (self.k + item["bm25_rank"])

        # Sort fused results by RRF score descending
        sorted_results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

        # Truncate to top_k and assign rank
        final_results = []
        for rank_idx, item in enumerate(sorted_results[:top_k]):
            item["rank"] = rank_idx + 1
            item["retrieval_method"] = "hybrid"
            final_results.append(item)
            
        return final_results
