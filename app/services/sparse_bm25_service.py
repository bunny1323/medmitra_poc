"""
Sparse BM25 embedding service using fastembed's Qdrant/bm25 model.
Generates sparse vectors for both indexing records and querying.
"""
from typing import List, Dict, Any
from app.core.logging import logger


class SparseBM25Service:
    """
    Wraps fastembed SparseTextEmbedding with Qdrant/bm25 model to produce
    sparse vectors compatible with Qdrant's sparse vector format.
    """

    MODEL_NAME = "Qdrant/bm25"

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from fastembed import SparseTextEmbedding
            logger.info(f"Loading sparse BM25 model: {self.MODEL_NAME}")
            self._model = SparseTextEmbedding(model_name=self.MODEL_NAME)
            logger.info("Sparse BM25 model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load sparse BM25 model: {e}")
            self._model = None

    def embed_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Embed a list of texts as sparse BM25 vectors.
        Returns list of {"indices": [...], "values": [...]}
        Falls back to empty vectors on failure.
        """
        self._load_model()
        if not self._model:
            return [{"indices": [], "values": []} for _ in texts]

        try:
            embeddings = list(self._model.embed(texts))
            result = []
            for emb in embeddings:
                result.append({
                    "indices": emb.indices.tolist(),
                    "values": emb.values.tolist()
                })
            return result
        except Exception as e:
            logger.error(f"Sparse embedding error: {e}")
            return [{"indices": [], "values": []} for _ in texts]

    def embed_query(self, query: str) -> Dict[str, Any]:
        """
        Embed a single query string as a sparse BM25 vector.
        """
        results = self.embed_texts([query])
        return results[0] if results else {"indices": [], "values": []}

    def embed_query_batch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Embed multiple queries."""
        return self.embed_texts(queries)

    def is_ready(self) -> bool:
        self._load_model()
        return self._model is not None
