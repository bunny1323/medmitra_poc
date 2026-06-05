import numpy as np
from typing import List, Union
from app.core.config import settings

class EmbeddingService:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL, device=settings.EMBEDDING_DEVICE)
        except Exception:
            pass

    def get_dense_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str): texts = [texts]
        if not self.model:
            # stable mocked fallback vectors
            results = []
            for t in texts:
                import hashlib
                seed = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                v = rng.standard_normal(768)
                v = v / np.linalg.norm(v)
                results.append(v.tolist())
            return results
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
