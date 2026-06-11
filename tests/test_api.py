import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core import config
from app.services import retrieval_service, llm_service

client = TestClient(app)

@pytest.fixture
def mock_retrieval_and_llm(monkeypatch):
    # Mock retrieval search
    def mock_search(query, top_k=5):
        return {
            "query": query,
            "results": [
                {
                    "rank": 1,
                    "id": "point-uuid",
                    "content": "Treatment guidelines for general cold and fever require plenty of rest.",
                    "relevance_score": 0.75,
                    "metadata": {
                        "page": 42,
                        "source_name": "Test Guideline",
                        "original_filename": "test.pdf"
                    }
                }
            ],
            "top_relevance": 0.75
        }
    monkeypatch.setattr(retrieval_service, "search", mock_search)
    monkeypatch.setattr(retrieval_service, "collection_exists", lambda name: True)
    
    # Mock LLM generation
    def mock_generate_response(query, retrieval_result):
        # Trigger safe refusal locally if matched
        safety_check = llm_service.check_unsafe_dosage_or_antibiotic(query)
        if safety_check:
            return safety_check
            
        return {
            "answer_mode": "retrieval_grounded",
            "answer": "Empathetic advice on cold and fever.",
            "home_cautions": ["Drink hot water", "Rest"],
            "model_used": "mock-llama",
            "source": "groq"
        }
    monkeypatch.setattr(llm_service, "generate_response", mock_generate_response)

def test_health_live():
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "live", "message": "MedMitra API is running"}

def test_health_ready_without_groq(monkeypatch):
    """The API remains ready because Groq is an optional dependency."""
    monkeypatch.setattr(llm_service, "is_groq_configured", lambda: False)

    response = client.get("/health/ready")

    assert response.status_code == 200
def test_emergency_query_flow():
    # Emergency checks run before security check? Wait, security check is standard dependency on `/query`,
    # so we still need correct header.
    headers = {"X-Internal-API-Key": config.INTERNAL_API_KEY}
    request_data = {
        "query": "I have severe chest pain and difficulty breathing",
        "top_k": 5
    }
    response = client.post("/api/v1/query", json=request_data, headers=headers)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["emergency_detected"] is True
    assert json_data["severity_index"] == "CRITICAL"
    assert "chest pain" in json_data["emergency_matches"]
    assert "difficulty breathing" in json_data["emergency_matches"]
    assert "urgent medical attention" in json_data["answer"]

def test_normal_symptom_query(mock_retrieval_and_llm):
    headers = {"X-Internal-API-Key": config.INTERNAL_API_KEY}
    request_data = {
        "query": "I have mild fever and cold",
        "top_k": 5
    }
    response = client.post("/api/v1/query", json=request_data, headers=headers)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["emergency_detected"] is False
    assert json_data["severity_index"] == "NORMAL"
    assert json_data["answer_mode"] == "retrieval_grounded"
    assert json_data["retrieval_relevance_score"] == 0.75
    assert json_data["retrieval_relevance_level"] == "MEDIUM"
    assert "cold and fever" in json_data["answer"]

def test_unsafe_dosage_request(mock_retrieval_and_llm):
    headers = {"X-Internal-API-Key": config.INTERNAL_API_KEY}
    request_data = {
        "query": "Tell me exactly how many antibiotic tablets I should take",
        "top_k": 5
    }
    response = client.post("/api/v1/query", json=request_data, headers=headers)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["answer_mode"] == "safe_refusal"
    assert "cannot provide personalized dosage" in json_data["answer"]
    assert json_data["severity_index"] == "NORMAL"
