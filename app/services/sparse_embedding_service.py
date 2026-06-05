from typing import List, Dict, Any, Union

class SparseEmbeddingService:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from fastembed import SparseTextEmbedding
            self.model = SparseTextEmbedding("Qdrant/bm25")
        except Exception:
            pass

    def get_sparse_embeddings(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        if isinstance(texts, str): texts = [texts]
        if not self.model:
            # fallback mock lexical sparse indexer
            results = []
            for t in texts:
                words = t.lower().split()
                freqs = {}
                for w in words:
                    idx = hash(w) % 100000
                    freqs[idx] = freqs.get(idx, 0.0) + 1.0
                total = sum(freqs.values())
                indices = sorted(freqs.keys())
                values = [freqs[idx] / total for idx in indices]
                results.append({"indices": indices, "values": values})
            return results
        embeddings = list(self.model.embed(texts))
        return [{"indices": emb.indices.tolist(), "values": emb.values.tolist()} for emb in embeddings]
