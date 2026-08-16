from __future__ import annotations

from collections import defaultdict
from contextvars import ContextVar
from typing import Any

REQUEST_ID = ContextVar("request_id", default="n/a")
TENANT_ID = ContextVar("tenant_id", default="default")


class AppMetrics:
    _counters: dict[str, int] = defaultdict(int)
    _latencies_ms: dict[str, list[float]] = defaultdict(list)

    @classmethod
    def increment(cls, name: str, value: int = 1) -> None:
        cls._counters[name] += value

    @classmethod
    def record_latency(cls, name: str, milliseconds: float) -> None:
        cls._latencies_ms[name].append(milliseconds)

    @classmethod
    def snapshot(cls) -> dict[str, Any]:
        return {
            "counters": dict(cls._counters),
            "latencies_ms": {
                key: {
                    "count": len(values),
                    "avg": sum(values) / len(values) if values else 0.0,
                    "max": max(values) if values else 0.0,
                }
                for key, values in cls._latencies_ms.items()
            },
        }
