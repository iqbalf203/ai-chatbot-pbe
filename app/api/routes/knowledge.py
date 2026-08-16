from __future__ import annotations

import importlib

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL, MAX_FILE_SIZE_BYTES, MONGODB_DATABASE, MONGODB_KNOWLEDGE_COLLECTION, MONGODB_URL
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.document_ingestion import DocumentIngestionService
from app.services.embedding_service import EmbeddingService

router = APIRouter(prefix="/api/admin/knowledge", tags=["knowledge"])


async def get_knowledge_repository() -> KnowledgeRepository:
    chat_app = importlib.import_module("demo_1_be.app")

    mongo_client = getattr(chat_app, "mongo_client", None)
    mongo_db = getattr(chat_app, "db", None)

    if mongo_client is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB is not configured or the application is not running.",
        )

    if mongo_db is None:
        raise HTTPException(
            status_code=503,
            detail="MongoDB database is not available.",
        )

    return KnowledgeRepository(mongo_client, MONGODB_DATABASE)


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def ingest_document(file: UploadFile = File(...)) -> dict[str, str | int]:
    try:
        repository = await get_knowledge_repository()
        embedding_service = EmbeddingService(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
            model=EMBEDDING_MODEL,
        )
        ingestion_service = DocumentIngestionService(
            repository=repository,
            embedding_service=embedding_service,
            max_file_size_bytes=MAX_FILE_SIZE_BYTES,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        return await ingestion_service.ingest(file)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge document ingestion failed: {exc}",
        ) from exc
