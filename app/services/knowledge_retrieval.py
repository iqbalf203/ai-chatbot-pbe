from __future__ import annotations

import logging

from app.core.config import KNOWLEDGE_SEARCH_LIMIT
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger("demo_1_be.knowledge")


class KnowledgeRetrievalService:
    def __init__(
        self,
        repository: KnowledgeRepository,
        embedding_service: EmbeddingService,
        limit: int = KNOWLEDGE_SEARCH_LIMIT,
    ):
        self.repository = repository
        self.embedding_service = embedding_service
        self.limit = limit

    async def retrieve(self, query: str) -> str:
        if not query or not query.strip():
            logger.info("Knowledge retrieval skipped because query is empty.")
            return ""

        trimmed_query = query.strip()

        try:
            logger.info("Generating embedding for retrieval query: %s", trimmed_query[:200])
            query_embedding = await self.embedding_service.generate_embedding(trimmed_query)
            logger.info("Embedding generated for retrieval query, length=%s", len(query_embedding))

            matches = await self.repository.find_similar_chunks(
                query_embedding=query_embedding,
                limit=self.limit,
            )
            logger.info("Knowledge retrieval returned %d matches.", len(matches))
        except Exception as exc:
            logger.warning("Knowledge retrieval failed: %s", exc, exc_info=True)
            raise

        if not matches:
            logger.info("No vector matches found for query: %s", trimmed_query[:200])
            return ""

        context_chunks = [
            f"Source: {match.get('filename', 'unknown')}\n{match.get('text', '').strip()}"
            for match in matches
            if match.get("text")
        ]

        context_text = "\n\n---\n\n".join(context_chunks)
        logger.info("Built retrieval context with %d chunks and %d chars.", len(context_chunks), len(context_text))
        return context_text
