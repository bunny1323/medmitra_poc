import pytest
from app.services.emergency_service import check_emergency, is_negated

def test_negation_detection():
    # Simple negation checks
    assert is_negated("I do not have chest pain", "chest pain") is True
    assert is_negated("patient denies difficulty breathing", "difficulty breathing") is True
    assert is_negated("without chest tightness", "chest tightness") is True
    assert is_negated("I have chest pain", "chest pain") is False
    assert is_negated("unconscious patient", "unconscious") is False

def test_emergency_query_match_all():
    # Multiple keywords match
    query = "I have severe chest pain and difficulty breathing"
    res = check_emergency(query)
    
    assert res is not None
    assert res["is_emergency"] is True
    assert res["severity"] == "CRITICAL"
    # Matches chest pain and difficulty breathing
    assert "chest pain" in res["matches"]
    assert "difficulty breathing" in res["matches"]

def test_emergency_with_negation():
    # Partially negated query
    query = "I have chest pain but no difficulty breathing"
    res = check_emergency(query)
    
    assert res is not None
    assert "chest pain" in res["matches"]
    assert "difficulty breathing" not in res["matches"]

def test_non_emergency_query():
    query = "I have a mild headache and feel a bit tired"
    res = check_emergency(query)
    assert res is None
