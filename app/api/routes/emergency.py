from fastapi import APIRouter, Depends
from app.schemas.emergency import EmergencyCheckRequest, EmergencyCheckResponse
from app.core.security import verify_internal_api_key
from app.services.container import container

router = APIRouter()

@router.post("/emergency-check", response_model=EmergencyCheckResponse, tags=["Emergency Check"])
async def check_emergency(payload: EmergencyCheckRequest, internal_key: None = Depends(verify_internal_api_key)):
    res = container.emergency_service.check_query(payload.text)
    
    # Extract matching phrases as simple strings
    matched_strings = [m.matched_phrase for m in res.matches] if hasattr(res, "matches") else []
    
    # Ensure standard warning message if emergency is detected
    message = res.message
    if res.is_emergency:
        message = "This may require urgent medical attention. Please contact emergency medical services or visit the nearest hospital immediately."
        
    return EmergencyCheckResponse(
        status="success",
        is_emergency=res.is_emergency,
        severity=res.severity,
        matches=matched_strings,
        message=message
    )

