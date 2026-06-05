"""
Disease Search Service
========================
Hybrid search pipeline for disease/symptom queries.

Flow:
  1. Emergency check (always first)
  2. Normalize query text
  3. RapidFuzz typo correction
  4. Qdrant hybrid search (dense MedCPT + sparse BM25) filtered by record_type=disease
  5. Fallback to local numpy/pickle when Qdrant is unavailable
  6. Compute matched symptoms and retrieval_relevance label
  7. Return informational results with disclaimer

IMPORTANT: This service returns general informational retrieval results only.
           It does NOT diagnose diseases or prescribe treatment.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.logging import logger
from app.models.enums import RelevanceLabel
from app.services.bm25_retriever import tokenize_text

DISCLAIMER = (
    "These are informational retrieval results and not a diagnosis. "
    "The data is sourced from Kaggle datasets (prototype_unverified) and has not been clinically validated. "
    "Consult a qualified healthcare professional for medical advice."
)


class DiseaseSearchService:
    """
    Hybrid disease search using Qdrant (primary) with local index fallback.
    """

    def __init__(self, typo_handler, emergency_detector, medcpt_retriever, rrf_fusion,
                 qdrant_service=None, sparse_bm25_service=None):
        self.typo_handler = typo_handler
        self.emergency_detector = emergency_detector
        self.medcpt_retriever = medcpt_retriever
        self.rrf_fusion = rrf_fusion
        self.qdrant_service = qdrant_service
        self.sparse_bm25_service = sparse_bm25_service

        # Local index fallback state
        self.bm25 = None
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: Optional[List[Dict]] = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Local index loading (fallback)
    # ------------------------------------------------------------------

    def load_local_index(self) -> bool:
        """Load local numpy + BM25 indexes as a fallback when Qdrant is unavailable."""
        if self._loaded:
            return True

        bm25_path = settings.disease_bm25_path
        emb_path = settings.disease_embeddings_path
        meta_path = os.path.join(settings.indexes_dir, "disease_metadata.json")

        if not all(os.path.exists(p) for p in [bm25_path, emb_path, meta_path]):
            logger.warning("Disease local index files not found — Qdrant is required or rebuild indexes.")
            return False

        try:
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            self.embeddings = np.load(emb_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self._loaded = True
            logger.info(f"DiseaseSearchService: loaded local index with {len(self.metadata)} records.")
            return True
        except Exception as e:
            logger.error(f"Error loading disease local index: {e}")
            return False

    # ------------------------------------------------------------------
    # Qdrant-backed search (primary)
    # ------------------------------------------------------------------

    def _search_qdrant(
        self,
        corrected_query: str,
        top_k: int,
        tokenized_query: List[str],
    ) -> List[Dict]:
        """Perform hybrid search via Qdrant, return normalised result dicts."""
        if not self.qdrant_service or not self.qdrant_service.is_connected():
            return []

        if not self.qdrant_service.check_collection_exists(settings.QDRANT_COLLECTION):
            logger.warning(f"Collection '{settings.QDRANT_COLLECTION}' not found. Run build_qdrant_index.py first.")
            return []

        try:
            # Dense embedding
            query_emb = self.medcpt_retriever.embed_query(corrected_query)
            dense_vec = query_emb.tolist()

            # Sparse BM25 embedding
            if self.sparse_bm25_service:
                sparse_vec = self.sparse_bm25_service.embed_query(corrected_query)
            else:
                sparse_vec = {"indices": [], "values": []}

            hits = self.qdrant_service.hybrid_search_by_type(
                collection_name=settings.QDRANT_COLLECTION,
                query_dense=dense_vec,
                query_sparse=sparse_vec,
                record_type="disease",
                limit=top_k,
            )

            query_words = set(tokenized_query)
            results = []
            for rank_idx, hit in enumerate(hits):
                p = hit["payload"]
                symptoms = p.get("symptoms", [])

                # Compute matched symptoms
                matched = []
                for sym in symptoms:
                    sym_words = set(tokenize_text(sym))
                    if sym_words & query_words:
                        matched.append(sym)

                results.append({
                    "rank": rank_idx + 1,
                    "condition_name": p.get("condition_name") or p.get("title", "Unknown"),
                    "matched_symptoms": matched,
                    "description": p.get("description", ""),
                    "precautions": p.get("precautions", []),
                    "source_name": p.get("source_name", ""),
                    "source_type": p.get("source_type", ""),
                    "dataset_slug": p.get("dataset_slug", ""),
                    "review_status": p.get("review_status", "prototype_unverified"),
                    "rrf_score": hit["rrf_score"],
                    "dense_score": hit.get("dense_score", 0.0),
                })

            return results

        except Exception as e:
            logger.error(f"Qdrant disease search error: {e}")
            return []

    # ------------------------------------------------------------------
    # Local fallback search
    # ------------------------------------------------------------------

    def _search_local(
        self,
        corrected_query: str,
        top_k: int,
        tokenized_query: List[str],
    ) -> List[Dict]:
        """Fallback: hybrid search using local numpy + BM25 indexes."""
        if not self.load_local_index():
            return []
        if not self.metadata or self.embeddings is None:
            return []

        # BM25
        bm25_scores = np.zeros(len(self.metadata))
        if tokenized_query:
            bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_rank_map = {}
        sorted_bm25 = np.argsort(bm25_scores)[::-1]
        for rank_idx, idx in enumerate(sorted_bm25):
            cid = self.metadata[idx].get("condition_id", str(idx))
            bm25_rank_map[cid] = {"rank": rank_idx + 1, "score": float(bm25_scores[idx])}

        # Dense (cosine)
        q_emb = self.medcpt_retriever.embed_query(corrected_query)
        norm_q = np.linalg.norm(q_emb)
        dense_rank_map = {}

        if norm_q > 0:
            norm_docs = np.linalg.norm(self.embeddings, axis=1)
            cosine = np.dot(self.embeddings, q_emb) / (norm_docs * norm_q + 1e-8)
            sorted_dense = np.argsort(cosine)[::-1]
            for rank_idx, idx in enumerate(sorted_dense):
                cid = self.metadata[idx].get("condition_id", str(idx))
                dense_rank_map[cid] = {"rank": rank_idx + 1, "score": float(cosine[idx])}

        # RRF fusion
        rrf_k = settings.RRF_K
        fused = []
        query_words = set(tokenized_query)

        for rec in self.metadata:
            cid = rec.get("condition_id", "")
            d = dense_rank_map.get(cid, {"rank": len(self.metadata) + 10, "score": 0.0})
            b = bm25_rank_map.get(cid, {"rank": len(self.metadata) + 10, "score": 0.0})

            # Only include records that meet the minimum relevance threshold
            if d["score"] < settings.DISEASE_SEARCH_MIN_RELEVANCE:
                continue

            rrf_score = 1.0 / (rrf_k + d["rank"]) + 1.0 / (rrf_k + b["rank"])
            symptoms = rec.get("symptoms", [])
            matched = [s for s in symptoms if set(tokenize_text(s)) & query_words]

            fused.append({
                "rank": 0,
                "condition_name": rec.get("condition_name", ""),
                "matched_symptoms": matched,
                "description": rec.get("description", ""),
                "precautions": rec.get("precautions", []),
                "source_name": rec.get("source_name", ""),
                "source_type": rec.get("source_type", ""),
                "dataset_slug": rec.get("dataset_slug", ""),
                "review_status": rec.get("review_status", "prototype_unverified"),
                "rrf_score": round(rrf_score, 6),
                "dense_score": round(d["score"], 4),
            })

        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        for i, item in enumerate(fused[:top_k]):
            item["rank"] = i + 1

        return fused[:top_k]

    # ------------------------------------------------------------------
    # Main search entry point
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        age_group: str = "adult",
        top_k: int = 5,
        include_full_text: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for disease / condition information matching the query.

        Returns:
            dict with keys: emergency_detected, normalized_query, results,
                            retrieval_relevance, disclaimer
        """
        # Step 1: Emergency check — always runs first
        try:
            emergency = self.emergency_detector.check_emergency(query)
        except Exception:
            emergency = type("obj", (object,), {"is_emergency": False, "message": ""})()

        if emergency.is_emergency:
            return {
                "emergency_detected": True,
                "emergency_message": getattr(emergency, "message", "Emergency signs detected."),
                "normalized_query": query,
                "results": [],
                "retrieval_relevance": RelevanceLabel.LOW,
                "disclaimer": DISCLAIMER,
            }

        # Step 2: Normalize text
        normalized_query = self.typo_handler.normalize_text(query)

        # Step 3: Typo correction
        corrected_query = self.typo_handler.correct_query(normalized_query)
        tokenized_query = tokenize_text(corrected_query)

        # Step 4: Qdrant search (primary)
        results = self._search_qdrant(corrected_query, top_k, tokenized_query)

        # Step 5: Local index fallback if Qdrant returns nothing
        if not results:
            logger.info("Qdrant unavailable or empty — falling back to local index.")
            results = self._search_local(corrected_query, top_k, tokenized_query)

        # Step 6: Compute retrieval relevance label
        max_dense = max((r.get("dense_score", 0.0) for r in results), default=0.0)
        if results:
            if max_dense >= 0.50:
                relevance = RelevanceLabel.HIGH
            elif max_dense >= settings.DISEASE_SEARCH_MIN_RELEVANCE or len(results) > 0:
                relevance = RelevanceLabel.MEDIUM
            else:
                relevance = RelevanceLabel.LOW
        else:
            relevance = RelevanceLabel.LOW

        return {
            "emergency_detected": False,
            "normalized_query": corrected_query,
            "results": results,
            "retrieval_relevance": relevance,
            "disclaimer": DISCLAIMER,
        }
