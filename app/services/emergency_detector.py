"""
EmergencyDetector — thin wrapper around EmergencyService.
Uses lazy import to avoid circular dependency with the container.
"""
from app.models.schemas import EmergencyCheckResponse


class EmergencyDetector:
    def __init__(self):
        self._service = None

    def _get_service(self):
        if self._service is None:
            from app.services.emergency_service import EmergencyService
            self._service = EmergencyService()
        return self._service

    def check_emergency(self, query: str) -> EmergencyCheckResponse:
        return self._get_service().check_query(query)
