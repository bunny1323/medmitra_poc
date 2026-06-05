import logging
import sys
import json
from typing import Any, Dict
from app.core.config import settings

class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "filename": record.filename,
            "line": record.lineno,
        }
        if hasattr(record, "extra_fields") and isinstance(record.extra_fields, dict):
            for key, val in record.extra_fields.items():
                if any(x in key.lower() for x in ["prescription", "text", "ocr", "file"]):
                    log_data[key] = "[REDACTED_FOR_SECURITY]"
                else:
                    log_data[key] = val
        return json.dumps(log_data)

def setup_logging() -> logging.Logger:
    logger = logging.getLogger(settings.APP_NAME)
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    if settings.APP_ENV.lower() == "production":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(module)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logging()
