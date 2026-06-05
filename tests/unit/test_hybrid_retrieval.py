import pytest
import numpy as np
import pickle
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.services.container import container
from app.services.rrf_fusion import RRFFusion
from app.services.bm25_retriever import tokenize_text
from app.models.schemas import HybridRagRequest, HybridRagResponse
from rank_bm25 import BM25Okapi

# Set up TestClient
client = TestClient(app)

# Sample medical chunks for testing
MOCK_CHUNKS = [
    {
        "chunk_id": "chunk_1",
        "text": "Paracetamol is used to treat fever and pain.",
        "source_title": "Guidelines on Paracetamol",
        "source_file": "paracetamol.pdf",
        "page_number": 5,
        "source_type": "official_guideline"
    },
    {
        "chunk_id": "chunk_2",
        "text": "Patients with fever and cough may have pneumonia.",
        "source_title": "Respiratory Infections Standard Treatement Workflow",
        "source_file": "respiratory_workflow.pdf",
        "page_number": 12,
        "source_type": "official_guideline"
    },
    {
        "chunk_id": "chunk_3",
        "text": "Sore throat is treated with warm saline gargles and rest.",
        "source_title": "ENT Standard Treatment Guideline",
        "source_file": "ent_guideline.pdf",
        "page_number": 3,
        "source_type": "official_guideline"
    }
]

@pytest.fixture
def mock_retrieval_indices():
    # Mock dense embeddings
    embeddings = np.array([
        [0.8, 0.1, 0.1],  # chunk_1
        [0.1, 0.8, 0.1],  # chunk_2
        [0.1, 0.1, 0.8]   # chunk_3
    ])
    
    # Mock BM25 index
    tokenized_corpus = [tokenize_text(c["text"]) for c in MOCK_CHUNKS]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return embeddings, MOCK_CHUNKS, bm25

def test_medcpt_dense_retrieval(mock_retrieval_indices):
    embeddings, metadata, _ = mock_retrieval_indices
    medcpt = container.medcpt_retriever
    
    medcpt.embeddings = embeddings
    medcpt.metadata = metadata
    
    # Mock embed_query to return a vector close to chunk_1
    medcpt.embed_query = MagicMock(return_value=np.array([0.9, 0.05, 0.05]))
    
    results = medcpt.retrieve("fever and pain", top_k=2)
    
    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk_1"
    assert results[0]["dense_rank"] == 1
    assert results[0]["dense_score"] > 0.9

def test_bm25_keyword_search(mock_retrieval_indices):
    _, metadata, bm25 = mock_retrieval_indices
    bm25_retriever = container.bm25_retriever
    
    bm25_retriever.bm25 = bm25
    bm25_retriever.metadata = metadata
    
    # Query for exact term "paracetamol"
    results = bm25_retriever.retrieve("paracetamol", top_k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] == "chunk_1"
    assert results[0]["bm25_rank"] == 1
    assert "paracetamol" in results[0]["text"].lower()

def test_rrf_score_calculation():
    rrf = RRFFusion(k=60)
    
    dense_results = [
        {"chunk_id": "chunk_1", "text": "...", "source_name": "A", "source_type": "official_guideline", "page_number": 1, "dense_rank": 1, "dense_score": 0.95},
        {"chunk_id": "chunk_2", "text": "...", "source_name": "B", "source_type": "official_guideline", "page_number": 2, "dense_rank": 2, "dense_score": 0.85}
    ]
    bm25_results = [
        {"chunk_id": "chunk_2", "text": "...", "source_name": "B", "source_type": "official_guideline", "page_number": 2, "bm25_rank": 1, "bm25_score": 1.5},
        {"chunk_id": "chunk_1", "text": "...", "source_name": "A", "source_type": "official_guideline", "page_number": 1, "bm25_rank": 2, "bm25_score": 1.2}
    ]
    
    fused = rrf.fuse(dense_results, bm25_results, top_k=2)
    assert len(fused) == 2
    assert fused[0]["rrf_score"] == pytest.approx(1/61 + 1/62)

def test_duplicate_chunk_fusion():
    rrf = RRFFusion(k=60)
    
    dense_results = [
        {"chunk_id": "chunk_1", "text": "Text 1", "source_name": "Source A", "source_type": "official_guideline", "page_number": 10, "dense_rank": 1, "dense_score": 0.9}
    ]
    bm25_results = [
        {"chunk_id": "chunk_1", "text": "Text 1", "source_name": "Source A", "source_type": "official_guideline", "page_number": 10, "bm25_rank": 1, "bm25_score": 2.5}
    ]
    
    fused = rrf.fuse(dense_results, bm25_results, top_k=1)
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "chunk_1"
    assert fused[0]["dense_rank"] == 1
    assert fused[0]["bm25_rank"] == 1
    assert fused[0]["rrf_score"] == pytest.approx(1/61 + 1/61)

def test_empty_query_validation():
    # Empty query should return empty list gracefully or fail validation
    # Pydantic request model checks min_length=1
    with pytest.raises(Exception):
        HybridRagRequest(query="", top_k=5)

def test_top_k_limits():
    # top_k ge=1, le=20 in model validation, so 25 should fail validation
    with pytest.raises(Exception):
        # Trigger validation
        HybridRagRequest(query="fever", top_k=25)

def test_emergency_symptom_detection():
    # Test cardiac emergency
    res_cardiac = container.emergency_detector.check_emergency("I have severe chest pain")
    assert res_cardiac.is_emergency is True
    assert any(m.category == "cardiac" for m in res_cardiac.matches)
    
    # Test respiratory emergency
    res_resp = container.emergency_detector.check_emergency("difficulty breathing")
    assert res_resp.is_emergency is True
    assert any(m.category == "respiratory" for m in res_resp.matches)
    
    # Test negation
    res_negated = container.emergency_detector.check_emergency("I have a cough but no chest pain")
    assert res_negated.is_emergency is False

def test_api_response_structure(mock_retrieval_indices):
    from app.core.security import verify_internal_api_key
    app.dependency_overrides[verify_internal_api_key] = lambda: None
    
    try:
        embeddings, metadata, bm25 = mock_retrieval_indices
        
        # Inject mocked search indices
        container.medcpt_retriever.embeddings = embeddings
        container.medcpt_retriever.metadata = metadata
        container.medcpt_retriever.embed_query = MagicMock(return_value=np.array([0.8, 0.1, 0.1]))
        
        container.bm25_retriever.bm25 = bm25
        container.bm25_retriever.metadata = metadata
        
        # Set vocabulary for typo handler to avoid empty vocab check
        container.typo_handler.vocab = {"paracetamol", "fever", "pain", "cough", "pneumonia", "sore", "throat", "saline", "gargles", "rest"}
        container.typo_handler._loaded = True
        
        response = client.post(
            "/api/v1/chat/rag",
            headers={"X-Internal-API-Key": "medmitra_internal_secure_key_123"},
            json={"query": "paracetamol", "top_k": 2}
        )
        
        assert response.status_code == 200
        res_json = response.json()
        assert res_json["status"] == "success"
        assert res_json["query"] == "paracetamol"
        assert res_json["emergency_detected"] is False
        assert res_json["emergency_message"] is None
        assert res_json["retrieval_method"] == "medcpt_bm25_rrf"
        assert "results" in res_json
        assert len(res_json["results"]) <= 2
        
        if len(res_json["results"]) > 0:
            first_result = res_json["results"][0]
            assert "rank" in first_result
            assert "chunk_id" in first_result
            assert "text" in first_result
            assert "source_name" in first_result
            assert "source_type" in first_result
            assert "page_number" in first_result
            assert "rrf_score" in first_result
            assert "retrieval_method" in first_result
            assert first_result["retrieval_method"] == "hybrid"
    finally:
        app.dependency_overrides.clear()

