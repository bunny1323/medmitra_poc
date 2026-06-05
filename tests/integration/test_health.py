from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_live():
    res = client.get("/health/live")
    assert res.status_code == 200
    assert res.json()["status"] == "alive"

def test_health_ready():
    res = client.get("/health/ready")
    assert res.status_code == 200
    assert "status" in res.json()
