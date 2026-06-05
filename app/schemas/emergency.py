from pydantic import BaseModel, Field
from typing import List, Optional
from app.models.enums import EmergencySeverity

class EmergencyCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

class EmergencyCheckResponse(BaseModel):
    status: str = "success"
    is_emergency: bool
    severity: Optional[EmergencySeverity] = None
    matches: List[str] = []
    message: str
