import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key in ("request_id", "conversation_id", "tenant_id", "component", "event", "status", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)

        return json.dumps(payload, default=str)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("demo_1_be")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(JsonFormatter())
    logger.addHandler(stream_handler)

    return logger


def log_event(level: int, message: str, **context) -> None:
    logger = logging.getLogger("demo_1_be")
    record = logger.makeRecord(
        logger.name,
        level,
        __file__,
        0,
        message,
        (),
        None,
    )
    record.extra = context
    logger.handle(record)
