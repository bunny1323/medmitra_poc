import pytest
from app.services import retrieval_service
from app.core import config

def test_empty_query_search():
    res = retrieval_service.search("")
    assert "error" in res
    assert res["top_relevance"] == 0.0
    assert len(res["results"]) == 0

def test_missing_collection_search(monkeypatch):
    # Mock collection_exists to return False
    monkeypatch.setattr(retrieval_service, "collection_exists", lambda name: False)
    
    res = retrieval_service.search("fever symptoms")
    assert "error" in res
    assert "not found" in res["error"]
    assert res["top_relevance"] == 0.0
    assert len(res["results"]) == 0

def test_retrieval_flow_keys(monkeypatch):
    # Mock collection_exists, get_dense_model, get_sparse_model, get_qdrant_client
    monkeypatch.setattr(retrieval_service, "collection_exists", lambda name: True)
    
    class MockDenseModel:
        def embed(self, texts):
            # Return list with mock embeddings
            return [[0.1] * 384 for _ in texts]
            
    class MockSparseModel:
        class QueryEmbedResult:
            class Indices:
                def tolist(self):
                    return [0, 1]
            class Values:
                def tolist(self):
                    return [0.5, 0.5]
            indices = Indices()
            values = Values()
            
        def query_embed(self, text):
            return [self.QueryEmbedResult()]
            
    monkeypatch.setattr(retrieval_service, "get_dense_model", lambda: MockDenseModel())
    monkeypatch.setattr(retrieval_service, "get_sparse_model", lambda: MockSparseModel())
    
    class MockPoint:
        def __init__(self, id, score, payload):
            self.id = id
            self.score = score
            self.payload = payload
            
    class MockQueryResult:
        points = [
            MockPoint(id="uuid-1", score=0.85, payload={"content": "match 1", "page": 10}),
            MockPoint(id="uuid-2", score=0.65, payload={"content": "match 2", "page": 12})
        ]
        
    class MockQdrantClient:
        def query_points(self, **kwargs):
            return MockQueryResult()
            
    monkeypatch.setattr(retrieval_service, "get_qdrant_client", lambda: MockQdrantClient())
    
    res = retrieval_service.search("asthma symptoms")
    
    assert "error" not in res or res["error"] is None
    assert res["top_relevance"] == 0.85
    assert len(res["results"]) == 2
    assert res["results"][0]["content"] == "match 1"
    assert res["results"][0]["relevance_score"] == 0.85
    assert res["results"][0]["metadata"]["page"] == 10
