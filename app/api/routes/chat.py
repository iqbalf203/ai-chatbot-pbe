from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api", tags=["chat"])


@router.get("/chat/health")
async def chat_health() -> dict[str, str]:
    return {"status": "ok"}
