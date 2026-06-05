import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(
    name="X-Internal-API-Key",
    auto_error=False,
    description="Internal API key used by the MedMitra backend service.",
)


def verify_internal_api_key(
    provided_key: str | None = Security(api_key_header),
) -> None:
    expected_key = os.getenv("INTERNAL_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="INTERNAL_API_KEY is not configured on the server.",
        )

    if not provided_key or not secrets.compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Internal-API-Key.",
        )
