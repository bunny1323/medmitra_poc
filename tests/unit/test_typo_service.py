import pytest
from app.services.typo_service import TypoService

@pytest.fixture
def typo_service():
    return TypoService()

def test_normalization(typo_service):
    assert typo_service.normalize_text("Rx\tParacetamol TDS!!!") == "Rx Paracetamol TDS"

def test_match_returns_not_configured(typo_service):
    res = typo_service.match_prescription_text("Paracetamol")
    assert res["status"] == "catalogue_not_configured"
    assert res["candidate_medicines"] == []
