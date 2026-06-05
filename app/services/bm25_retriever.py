import os
import pickle
import re
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.logging import logger

def tokenize_text(text: str) -> List[str]:
    # Normalizes to lowercase and extracts alphanumeric tokens
    return re.findall(r"\b\w+\b", text.lower())

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.metadata = None

    def load_index(self) -> bool:
        index_path = settings.bm25_index_path
        meta_path = settings.chunk_metadata_path
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.warning(f"BM25 index files not found: {index_path} or {meta_path}")
            return False
            
        try:
            with open(index_path, "rb") as f:
                self.bm25 = pickle.load(f)
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded BM25 index with {len(self.metadata)} documents.")
            return True
        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
            return False

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.bm25 is None or self.metadata is None:
            if not self.load_index():
                return []
                
        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []
            
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort and get top-k
        top_k = min(top_k, len(scores))
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank_idx, idx in enumerate(top_indices):
            meta = self.metadata[idx]
            results.append({
                "chunk_id": meta["chunk_id"],
                "text": meta["text"],
                "source_name": meta["source_title"],
                "source_type": meta.get("source_type", "official_guideline"),
                "page_number": meta["page_number"],
                "bm25_rank": rank_idx + 1,
                "bm25_score": float(scores[idx])
            })
        return results
