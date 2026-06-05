import pytest
from app.services.emergency_service import EmergencyService

@pytest.fixture
def emergency_service():
    return EmergencyService()

def test_chest_pain_emergency(emergency_service):
    response = emergency_service.check_query("chest pain")
    assert response.is_emergency is True
    assert any(m.category == "cardiac" for m in response.matches)

def test_severe_chest_pain_and_cannot_breathe(emergency_service):
    response = emergency_service.check_query("severe chest pain and cannot breathe")
    assert response.is_emergency is True
    categories = {m.category for m in response.matches}
    assert "cardiac" in categories
    assert "respiratory" in categories

def test_negated_chest_pain(emergency_service):
    response = emergency_service.check_query("I have cough but no chest pain")
    assert response.is_emergency is False
    assert len(response.matches) == 0

def test_mild_cough_normal(emergency_service):
    response = emergency_service.check_query("mild cough since yesterday")
    assert response.is_emergency is False
    assert len(response.matches) == 0

def test_child_unable_to_drink(emergency_service):
    response = emergency_service.check_query("child is unable to drink")
    assert response.is_emergency is True
    assert any(m.category == "pediatric_danger_sign" for m in response.matches)

def test_self_harm(emergency_service):
    response = emergency_service.check_query("I want to harm myself")
    assert response.is_emergency is True
    assert any(m.category == "self_harm" for m in response.matches)
