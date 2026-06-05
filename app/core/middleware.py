import time
import uuid
from typing import Callable, Dict, Tuple
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge
from app.core.config import settings
from app.core.logging import logger

REQUEST_COUNT = Counter("medmitra_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("medmitra_request_latency_seconds", "HTTP latency", ["method", "endpoint"])
EMERGENCY_TRIGGER_COUNT = Counter("medmitra_emergency_triggers_total", "Emergency keywords triggered", ["category"])
RETRIEVAL_LATENCY = Histogram("medmitra_retrieval_latency_seconds", "Qdrant latency")
LLM_LATENCY = Histogram("medmitra_llm_latency_seconds", "LLM latency")
QDRANT_ERROR_COUNT = Counter("medmitra_qdrant_errors_total", "Qdrant errors")
OCR_REQUEST_COUNT = Counter("medmitra_ocr_requests_total", "OCR requests")
INGESTION_PROGRESS = Gauge("medmitra_ingestion_progress_percent", "Ingestion percent")
INDEXED_POINT_COUNT = Gauge("medmitra_indexed_point_count", "Indexed points count")

ip_request_history: Dict[str, list] = {}

def is_rate_limited(client_ip: str) -> Tuple[bool, int]:
    now = time.time()
    if client_ip not in ip_request_history:
        ip_request_history[client_ip] = [now]
        return False, 59
    timestamps = [ts for ts in ip_request_history[client_ip] if now - ts < 60.0]
    ip_request_history[client_ip] = timestamps
    if len(timestamps) >= 60:
        return True, 0
    ip_request_history[client_ip].append(now)
    return False, 60 - len(ip_request_history[client_ip])

async def process_request_middleware(request: Request, call_next: Callable) -> Response:
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = req_id
    client_ip = request.client.host if request.client else "unknown"
    
    if not any(request.url.path.endswith(x) for x in ["/health/live", "/health/ready", "/metrics"]):
        limited, remaining = is_rate_limited(client_ip)
        if limited:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"status": "error", "code": "RATE_LIMIT_EXCEEDED", "message": "Too many requests. Please try again later."},
                headers={"Retry-After": "60", "X-RateLimit-Limit": "60", "X-RateLimit-Remaining": "0"}
            )
    else:
        remaining = 60
        
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    
    path = request.url.path
    if not any(path.endswith(x) for x in ["/metrics", "/health/live", "/health/ready"]):
        REQUEST_COUNT.labels(method=request.method, endpoint=path, status=response.status_code).inc()
        REQUEST_LATENCY.labels(method=request.method, endpoint=path).observe(process_time)
    return response

def setup_cors(app) -> None:
    origins = ["http://localhost", "http://localhost:3000", "http://localhost:8000", "http://127.0.0.1"]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])
