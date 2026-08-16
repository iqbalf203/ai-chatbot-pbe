from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import UploadFile

from app.core.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_FILE_SIZE_BYTES
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.document_parser import (
    extract_text,
    split_text_into_chunks,
    validate_file,
)
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger("demo_1_be.ingestion")


class DocumentIngestionService:
    def __init__(
        self,
        repository: KnowledgeRepository,
        embedding_service: EmbeddingService,
        max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.repository = repository
        self.embedding_service = embedding_service
        self.max_file_size_bytes = max_file_size_bytes
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def current_timestamp() -> datetime:
        return datetime.now(timezone.utc)

    async def ingest(self, file: UploadFile, tenant_id: str | None = None) -> dict[str, Any]:
        validate_file(file, max_size_bytes=self.max_file_size_bytes)

        try:
            text = extract_text(file)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Failed to extract text from document: {exc}") from exc

        if not text or not text.strip():
            raise ValueError("The uploaded document is empty or contains no readable text.")

        chunks = split_text_into_chunks(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        document_id = str(uuid.uuid4())
        saved_chunks: list[dict[str, Any]] = []

        for chunk_index, chunk in enumerate(chunks):
            try:
                embedding = await self.embedding_service.generate_embedding(chunk)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Embedding failed for chunk {chunk_index}: {exc}"
                ) from exc

            saved_chunks.append(
                {
                    "document_id": document_id,
                    "filename": file.filename,
                    "chunk_index": chunk_index,
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": {
                        "source_file_type": file.filename.rsplit(".", 1)[-1].lower(),
                        "chunk_size": self.chunk_size,
                        "chunk_overlap": self.chunk_overlap,
                        "tenant_id": tenant_id or "default",
                        "status": "active",
                        "created_at": self.current_timestamp().isoformat(),
                    },
                    "tenant_id": tenant_id or "default",
                    "status": "active",
                    "created_at": self.current_timestamp(),
                }
            )

        await self.repository.ensure_indexes()
        await self.repository.insert_chunks(saved_chunks)

        asyncio.create_task(
            self._background_ingestion_job(
                document_id=document_id,
                filename=file.filename,
                chunks_created=len(saved_chunks),
                tenant_id=tenant_id,
            )
        )

        return {
            "document_id": document_id,
            "filename": file.filename,
            "chunks_created": len(saved_chunks),
            "status": "completed",
        }

    async def _background_ingestion_job(
        self,
        document_id: str,
        filename: str | None,
        chunks_created: int,
        tenant_id: str | None,
    ) -> None:
        await asyncio.sleep(0)
        logger.info(
            "Background ingestion job completed",
            extra={
                "component": "ingestion_job",
                "status": "completed",
                "document_id": document_id,
                "filename": filename,
                "tenant_id": tenant_id or "default",
                "chunks_created": chunks_created,
            },
        )
