from fastapi import APIRouter, Depends, status
from app.core.security import verify_internal_api_key

router = APIRouter()

@router.post("/inventory/sync", status_code=status.HTTP_501_NOT_IMPLEMENTED, tags=["Inventory Integration"])
async def sync_inventory(internal_key: None = Depends(verify_internal_api_key)):
    return {"status": "external_backend_required", "message": "Inventory synchronization must be implemented by the main backend team after the application database and integration contract are finalized."}

