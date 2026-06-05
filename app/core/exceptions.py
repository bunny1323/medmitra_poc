from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from app.core.logging import logger

class MedMitraException(Exception):
    def __init__(self, message: str, status_code: int = 500, code: str = "INTERNAL_ERROR"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code

class MedicalSafetyException(MedMitraException):
    def __init__(self, message: str, code: str = "SAFETY_LIMITATION"):
        super().__init__(message, status_code=status.HTTP_400_BAD_REQUEST, code=code)

class DatabaseConnectionException(MedMitraException):
    def __init__(self, message: str = "Vector database is currently unavailable."):
        super().__init__(message, status_code=status.HTTP_503_SERVICE_UNAVAILABLE, code="DATABASE_UNAVAILABLE")

class ExternalServiceException(MedMitraException):
    def __init__(self, message: str, code: str = "EXTERNAL_SERVICE_FAILURE"):
        super().__init__(message, status_code=status.HTTP_502_BAD_GATEWAY, code=code)

def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(MedMitraException)
    async def medmitra_exception_handler(request: Request, exc: MedMitraException):
        logger.error(f"MedMitra Exception: {exc.message}", extra={"extra_fields": {"code": exc.code}})
        return JSONResponse(status_code=exc.status_code, content={"status": "error", "code": exc.code, "message": exc.message})

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content={"status": "error", "code": "HTTP_ERROR", "message": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        errors = [{"field": " -> ".join(str(x) for x in err.get("loc", [])), "message": err.get("msg")} for err in exc.errors()]
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"status": "error", "code": "VALIDATION_ERROR", "message": "Input validation failed.", "details": errors})

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.critical(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"status": "error", "code": "SERVER_ERROR", "message": "An unexpected error occurred. Please contact support."})
