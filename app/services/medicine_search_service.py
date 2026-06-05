"""
Medicine Search Service
========================
Hybrid search pipeline for medicine / drug information queries.

Flow:
  1. Normalize query text
  2. RapidFuzz typo correction (medicine name + alias vocabulary)
  3. Exact name / alias match via Qdrant scroll
  4. Qdrant hybrid search (dense MedCPT + sparse BM25) filtered by record_type=medicine
  5. Fallback to local numpy/pickle when Qdrant is unavailable
  6. Inject antibiotic prescription warnings
  7. Return with disclaimer

IMPORTANT: This service returns general informational retrieval results only.
           It does NOT prescribe medicines, recommend dosages, or suggest
           self-medication. Antibiotics require a doctor's prescription.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from rapidfuzz import process, fuzz
from app.core.config import settings
from app.core.logging import logger
from app.models.enums import RelevanceLabel
from app.services.bm25_retriever import tokenize_text

DISCLAIMER = (
    "This is general medicine information. Do not start, stop or change medication "
    "without advice from a qualified healthcare professional. "
    "Data sourced from Kaggle datasets (prototype_unverified) and has not been clinically validated."
)

ANTIBIOTIC_WARNING = (
    "This is a prescription antibiotic. Do not use without a doctor's prescription. "
    "Misuse contributes to antimicrobial resistance."
)

_ANTIBIOTIC_KEYWORDS = {"antibiotic", "antimicrobial", "antibacterial", "anti-infective"}


def _inject_antibiotic_warning(category: str, existing_warnings: List[str]) -> List[str]:
    """Ensure antibiotic warning is present when the category indicates it."""
    category_lower = category.lower()
    if any(kw in category_lower for kw in _ANTIBIOTIC_KEYWORDS):
        if not any("antibiotic" in w.lower() for w in existing_warnings):
            return existing_warnings + [ANTIBIOTIC_WARNING]
    return existing_warnings


class MedicineSearchService:
    """
    Hybrid medicine search using Qdrant (primary) with local index fallback.
    """

    def __init__(self, typo_handler, medcpt_retriever, rrf_fusion,
                 qdrant_service=None, sparse_bm25_service=None):
        self.typo_handler = typo_handler
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
        if self._loaded:
            return True

        bm25_path = settings.medicine_bm25_path
        emb_path = settings.medicine_embeddings_path
        meta_path = os.path.join(settings.indexes_dir, "medicine_metadata.json")

        if not all(os.path.exists(p) for p in [bm25_path, emb_path, meta_path]):
            logger.warning("Medicine local index files not found — Qdrant is required or rebuild indexes.")
            return False

        try:
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            self.embeddings = np.load(emb_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self._loaded = True
            logger.info(f"MedicineSearchService: loaded local index with {len(self.metadata)} records.")
            return True
        except Exception as e:
            logger.error(f"Error loading medicine local index: {e}")
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_drug_vocab(self, metadata: List[Dict]) -> Dict[str, int]:
        """Build a lower-cased drug name/alias → index mapping."""
        vocab = {}
        for idx, rec in enumerate(metadata):
            name = rec.get("medicine_name", "").lower().strip()
            if name:
                vocab[name] = idx
            for alias in rec.get("aliases", []):
                a = alias.lower().strip()
                if a:
                    vocab[a] = idx
        return vocab

    def _format_result(self, rec: Dict, rank: int, match_type: str, rrf_score: float) -> Dict:
        category = rec.get("category", "")
        warnings = _inject_antibiotic_warning(category, rec.get("warnings", []))
        return {
            "rank": rank,
            "medicine_name": rec.get("medicine_name", ""),
            "generic_name": rec.get("generic_name", ""),
            "category": category,
            "uses": rec.get("uses", []),
            "side_effects": rec.get("side_effects", []),
            "warnings": warnings,
            "mechanism_of_action": rec.get("mechanism_of_action", ""),
            "salt_composition": rec.get("salt_composition", ""),
            "source_name": rec.get("source_name", ""),
            "source_type": rec.get("source_type", ""),
            "dataset_slug": rec.get("dataset_slug", ""),
            "review_status": rec.get("review_status", "prototype_unverified"),
            "match_type": match_type,
            "rrf_score": round(rrf_score, 6),
        }

    # ------------------------------------------------------------------
    # Qdrant-backed search (primary)
    # ------------------------------------------------------------------

    def _search_qdrant(
        self,
        corrected_query: str,
        top_k: int,
        original_query_words: List[str],
    ) -> List[Dict]:
        if not self.qdrant_service or not self.qdrant_service.is_connected():
            return []

        if not self.qdrant_service.check_collection_exists(settings.QDRANT_COLLECTION):
            logger.warning(f"Collection '{settings.QDRANT_COLLECTION}' not found. Run build_qdrant_index.py first.")
            return []

        try:
            query_emb = self.medcpt_retriever.embed_query(corrected_query)
            dense_vec = query_emb.tolist()

            if self.sparse_bm25_service:
                sparse_vec = self.sparse_bm25_service.embed_query(corrected_query)
            else:
                sparse_vec = {"indices": [], "values": []}

            hits = self.qdrant_service.hybrid_search_by_type(
                collection_name=settings.QDRANT_COLLECTION,
                query_dense=dense_vec,
                query_sparse=sparse_vec,
                record_type="medicine",
                limit=top_k,
            )

            query_words_lower = set(w.lower() for w in original_query_words)
            results = []
            for rank_idx, hit in enumerate(hits):
                p = hit["payload"]
                med_name = (p.get("medicine_name") or "").lower()
                aliases = [a.lower() for a in p.get("aliases", [])]

                # Determine match type
                if med_name in query_words_lower or any(a in query_words_lower for a in aliases):
                    match_type = "exact_name_match"
                elif any(med_name in w or w in med_name for w in query_words_lower):
                    match_type = "fuzzy_name_match"
                else:
                    match_type = "semantic_match"

                category = p.get("category", "")
                warnings = _inject_antibiotic_warning(category, p.get("warnings", []))

                results.append({
                    "rank": rank_idx + 1,
                    "medicine_name": p.get("medicine_name", ""),
                    "generic_name": p.get("generic_name", ""),
                    "category": category,
                    "uses": p.get("uses", []),
                    "side_effects": p.get("side_effects", []),
                    "warnings": warnings,
                    "mechanism_of_action": p.get("mechanism_of_action", ""),
                    "salt_composition": p.get("salt_composition", ""),
                    "source_name": p.get("source_name", ""),
                    "source_type": p.get("source_type", ""),
                    "dataset_slug": p.get("dataset_slug", ""),
                    "review_status": p.get("review_status", "prototype_unverified"),
                    "match_type": match_type,
                    "rrf_score": hit["rrf_score"],
                    "dense_score": hit.get("dense_score", 0.0),
                })

            return results

        except Exception as e:
            logger.error(f"Qdrant medicine search error: {e}")
            return []

    # ------------------------------------------------------------------
    # Local fallback search
    # ------------------------------------------------------------------

    def _search_local(
        self,
        corrected_query: str,
        original_query: str,
        top_k: int,
        allow_typo_correction: bool,
    ) -> List[Dict]:
        if not self.load_local_index():
            return []

        drug_vocab = self._build_drug_vocab(self.metadata)
        drug_names = list(drug_vocab.keys())

        # Expand typo handler vocab with drug names for correction
        if allow_typo_correction:
            self.typo_handler.vocab.update(drug_names)

        original_words = tokenize_text(original_query)
        corrected_words = tokenize_text(corrected_query)

        exact_matches = []
        fuzzy_matches = []
        seen_idx = set()

        for word in corrected_words:
            if word in drug_vocab:
                idx = drug_vocab[word]
                match_type = "exact_name_match" if word in [w.lower() for w in original_words] else "fuzzy_name_match"
                if idx not in seen_idx:
                    exact_matches.append((idx, match_type, 1.0))
                    seen_idx.add(idx)
            else:
                match = process.extractOne(word, drug_names, scorer=fuzz.ratio, score_cutoff=80.0)
                if match:
                    idx = drug_vocab[match[0]]
                    if idx not in seen_idx:
                        fuzzy_matches.append((idx, "fuzzy_name_match", match[1] / 100.0))
                        seen_idx.add(idx)

        results = []
        rank = 1

        # Exact + fuzzy name matches first
        for idx, match_type, score in exact_matches + sorted(fuzzy_matches, key=lambda x: x[2], reverse=True):
            rec = self.metadata[idx]
            results.append(self._format_result(rec, rank, match_type, score))
            rank += 1

        # Semantic fallback (BM25 + MedCPT RRF)
        bm25_rank_map: Dict[str, Dict] = {}
        if corrected_words:
            bm25_scores = self.bm25.get_scores(corrected_words)
            sorted_bm25 = np.argsort(bm25_scores)[::-1]
            for r_idx, idx in enumerate(sorted_bm25):
                mid = self.metadata[idx].get("medicine_id") or self.metadata[idx].get("record_id", str(idx))
                bm25_rank_map[mid] = {"rank": r_idx + 1, "score": float(bm25_scores[idx])}

        q_emb = self.medcpt_retriever.embed_query(corrected_query)
        norm_q = np.linalg.norm(q_emb)
        dense_rank_map: Dict[str, Dict] = {}
        if norm_q > 0:
            norm_docs = np.linalg.norm(self.embeddings, axis=1)
            cosine = np.dot(self.embeddings, q_emb) / (norm_docs * norm_q + 1e-8)
            sorted_dense = np.argsort(cosine)[::-1]
            for r_idx, idx in enumerate(sorted_dense):
                mid = self.metadata[idx].get("medicine_id") or self.metadata[idx].get("record_id", str(idx))
                dense_rank_map[mid] = {"rank": r_idx + 1, "score": float(cosine[idx])}

        rrf_k = settings.RRF_K
        semantic_items = []
        for idx, rec in enumerate(self.metadata):
            if idx in seen_idx:
                continue
            mid = rec.get("medicine_id") or rec.get("record_id", str(idx))
            d = dense_rank_map.get(mid, {"rank": len(self.metadata) + 10, "score": 0.0})
            b = bm25_rank_map.get(mid, {"rank": len(self.metadata) + 10, "score": 0.0})

            if d["score"] < settings.MEDICINE_SEARCH_MIN_RELEVANCE:
                continue

            rrf_score = 1.0 / (rrf_k + d["rank"]) + 1.0 / (rrf_k + b["rank"])
            semantic_items.append((idx, rrf_score, d["score"]))

        semantic_items.sort(key=lambda x: x[1], reverse=True)
        for idx, rrf_score, _ in semantic_items:
            if rank > top_k:
                break
            rec = self.metadata[idx]
            results.append(self._format_result(rec, rank, "semantic_match", rrf_score))
            rank += 1

        return results[:top_k]

    # ------------------------------------------------------------------
    # Main search entry point
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        allow_typo_correction: bool = True,
        include_full_text: bool = False,
    ) -> Dict[str, Any]:
        """
        Search for medicine information matching the query.

        Returns:
            dict with keys: corrected_query, results, retrieval_relevance, disclaimer
        """
        # Step 1: Normalize
        normalized_query = self.typo_handler.normalize_text(query)

        # Step 2: Typo correction
        corrected_query = normalized_query
        if allow_typo_correction:
            corrected_query = self.typo_handler.correct_query(normalized_query)

        original_words = tokenize_text(normalized_query)

        # Step 3: Qdrant (primary)
        results = self._search_qdrant(corrected_query, top_k, original_words)

        # Step 4: Local fallback if needed
        if not results:
            logger.info("Qdrant unavailable or empty — falling back to local medicine index.")
            results = self._search_local(corrected_query, normalized_query, top_k, allow_typo_correction)

        # Step 5: Compute relevance label
        if results:
            has_name_match = any(r["match_type"] in ("exact_name_match", "fuzzy_name_match") for r in results)
            if has_name_match:
                relevance = RelevanceLabel.HIGH
            else:
                max_dense = max((r.get("dense_score", 0.0) for r in results), default=0.0)
                if max_dense >= 0.50:
                    relevance = RelevanceLabel.HIGH
                elif max_dense >= settings.MEDICINE_SEARCH_MIN_RELEVANCE or len(results) > 0:
                    relevance = RelevanceLabel.MEDIUM
                else:
                    relevance = RelevanceLabel.LOW
        else:
            relevance = RelevanceLabel.LOW

        return {
            "corrected_query": corrected_query,
            "results": results,
            "retrieval_relevance": relevance,
            "disclaimer": DISCLAIMER,
        }
