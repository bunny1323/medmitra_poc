import pytest
from unittest.mock import MagicMock
from app.services.rag_service import RagService
from app.services.emergency_service import EmergencyService
from app.services.query_classifier import QueryClassifier
from app.models.schemas import ChatRagRequest
from app.models.enums import AgeGroup

@pytest.fixture
def rag_service():
    return RagService(
        EmergencyService(),
        QueryClassifier(),
        MagicMock(),
        MagicMock()
    )

def test_emergency_bypasses_rag(rag_service):
    req = ChatRagRequest(query="I have severe chest pain spreads to left arm", age_group=AgeGroup.ADULT)
    res = rag_service.process_chat_rag(req)
    assert res.is_emergency is True
    assert not rag_service.retrieval_service.retrieve.called

def test_unsupported_query(rag_service):
    req = ChatRagRequest(query="How to fix laptop?", age_group=AgeGroup.ADULT)
    res = rag_service.process_chat_rag(req)
    assert res.is_emergency is False
    assert "unrelated" in res.answer.summary
