from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.core import config

API_KEY_HEADER = APIKeyHeader(name="X-Internal-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is missing"
        )
    if api_key != config.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key
