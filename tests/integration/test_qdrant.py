import pytest
from unittest.mock import MagicMock
from app.services.qdrant_service import QdrantService
from app.core.exceptions import DatabaseConnectionException

def test_qdrant_unreachable():
    service = QdrantService()
    service.client = None
    assert service.is_connected() is False
    with pytest.raises(DatabaseConnectionException):
        service.upload_points("test", [], [], [])

def test_qdrant_recreate_collection():
    service = QdrantService()
    service.client = MagicMock()
    service.recreate_collection("col")
    assert service.client.recreate_collection.called
