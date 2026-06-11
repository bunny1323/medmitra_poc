import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core import config

client = TestClient(app)

def test_missing_api_key():
    # Calling secure endpoint without the key header should fail
    response = client.post("/api/v1/query", json={"query": "test query", "top_k": 3})
    assert response.status_code == 401
    assert "API Key is missing" in response.json()["detail"]

def test_wrong_api_key():
    # Calling secure endpoint with an invalid key should fail
    headers = {"X-Internal-API-Key": "wrong_secret_key"}
    response = client.post("/api/v1/query", json={"query": "test query", "top_k": 3}, headers=headers)
    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]

def test_correct_api_key():
    # Calling secure endpoint with correct key should pass security (might fail with database errors, but security should accept it)
    # So we check that it doesn't return 401
    headers = {"X-Internal-API-Key": config.INTERNAL_API_KEY}
    response = client.post("/api/v1/query", json={"query": "I have severe chest pain", "top_k": 3}, headers=headers)
    
    # Since chest pain is an emergency, it returns immediately (200 OK) without calling Qdrant or Groq
    assert response.status_code == 200
    assert response.json()["emergency_detected"] is True
