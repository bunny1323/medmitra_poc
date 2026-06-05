import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.core.logging import logger

class MedCPTRetriever:
    def __init__(self):
        self.query_tokenizer = None
        self.query_model = None
        self.article_tokenizer = None
        self.article_model = None
        
        self.embeddings = None
        self.metadata = None
        
    def _load_query_model(self):
        if self.query_model is None:
            logger.info("Loading MedCPT Query Encoder model...")
            self.query_tokenizer = AutoTokenizer.from_pretrained(settings.MEDCPT_QUERY_MODEL)
            self.query_model = AutoModel.from_pretrained(settings.MEDCPT_QUERY_MODEL)
            logger.info("MedCPT Query Encoder loaded successfully.")

    def _load_article_model(self):
        if self.article_model is None:
            logger.info("Loading MedCPT Article Encoder model...")
            self.article_tokenizer = AutoTokenizer.from_pretrained(settings.MEDCPT_ARTICLE_MODEL)
            self.article_model = AutoModel.from_pretrained(settings.MEDCPT_ARTICLE_MODEL)
            logger.info("MedCPT Article Encoder loaded successfully.")

    def load_index(self) -> bool:
        """Loads precomputed dense embeddings and metadata from disk."""
        emb_path = settings.medcpt_embeddings_path
        meta_path = settings.chunk_metadata_path
        
        if not os.path.exists(emb_path) or not os.path.exists(meta_path):
            logger.warning(f"Dense index files not found: {emb_path} or {meta_path}")
            return False
            
        try:
            self.embeddings = np.load(emb_path)
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded {len(self.metadata)} embeddings and metadata successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading dense index: {e}")
            return False

    def embed_query(self, query: str) -> np.ndarray:
        self._load_query_model()
        with torch.no_grad():
            encoded = self.query_tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            )
            # Extracted CLS token embedding
            embeds = self.query_model(**encoded).last_hidden_state[:, 0, :]
            return embeds.cpu().numpy()[0]

    def embed_articles(self, articles: List[List[str]]) -> np.ndarray:
        """
        Embed list of articles using Article Encoder.
        Each article is [title, text] or a single string.
        """
        self._load_article_model()
        embeddings = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(articles), batch_size):
                batch = articles[i : i + batch_size]
                # If batch elements are list of [title, text], tokenize title and text
                if isinstance(batch[0], list):
                    titles = [x[0] for x in batch]
                    texts = [x[1] for x in batch]
                    encoded = self.article_tokenizer(
                        titles,
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                else:
                    encoded = self.article_tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt"
                    )
                
                embeds = self.article_model(**encoded).last_hidden_state[:, 0, :]
                embeddings.append(embeds.cpu().numpy())
        return np.concatenate(embeddings, axis=0)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.embeddings is None or self.metadata is None:
            if not self.load_index():
                return []
                
        # Get query embedding
        q_emb = self.embed_query(query)
        
        # Calculate cosine similarity
        norm_q = np.linalg.norm(q_emb)
        norm_docs = np.linalg.norm(self.embeddings, axis=1)
        
        # Guard against zero-division
        if norm_q == 0:
            return []
        
        dot_product = np.dot(self.embeddings, q_emb)
        scores = dot_product / (norm_docs * norm_q)
        
        # Get top-k indices
        top_k = min(top_k, len(scores))
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
                "dense_rank": rank_idx + 1,
                "dense_score": float(scores[idx])
            })
        return results
