from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from typing import Optional
from app.models.schemas import PrescriptionMatchRequest, PrescriptionMatchResponse
from app.core.security import verify_internal_api_key
from app.services.container import container
from app.core.config import settings

router = APIRouter()

@router.post("/prescriptions/upload", tags=["Prescription Processing"])
async def upload_prescription(
    file: UploadFile = File(...),
    language_hint: Optional[str] = Form(None),
    require_manual_review: bool = Form(True),
    internal_key: None = Depends(verify_internal_api_key)
):
    ext = file.filename.split(".")[-1].lower() if file.filename else ""
    if ext not in ["pdf", "png", "jpg", "jpeg"]:
        raise HTTPException(status_code=400, detail="Unsupported format.")
        
    body_bytes = await file.read()
    if len(body_bytes) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large.")
        
    return container.ocr_service.process_upload(body_bytes, file.filename or "file.pdf", file.content_type or "application/pdf", language_hint, require_manual_review)

@router.post("/prescriptions/match-text", response_model=PrescriptionMatchResponse, tags=["Prescription Processing"])
async def match_prescription_text(payload: PrescriptionMatchRequest, internal_key: None = Depends(verify_internal_api_key)):
    return container.typo_service.match_prescription_text(payload.text, payload.manual_review_confirmed)

