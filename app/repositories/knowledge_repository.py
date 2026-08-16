from __future__ import annotations

import logging
from typing import Any

from pymongo import ASCENDING

from app.core.config import (
    MONGODB_DATABASE,
    MONGODB_KNOWLEDGE_COLLECTION,
    MONGODB_KNOWLEDGE_INDEX_NAME,
)

logger = logging.getLogger("demo_1_be.knowledge_repository")

VECTOR_SEARCH_INDEX_DEFINITION = {
    "name": MONGODB_KNOWLEDGE_INDEX_NAME,
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1536,
                "similarity": "cosine",
            }
        ]
    },
}


class KnowledgeRepository:
    def __init__(self, mongo_client: Any, db_name: str = MONGODB_DATABASE):
        self.client = mongo_client
        self.db = mongo_client[db_name]
        self.collection = self.db[MONGODB_KNOWLEDGE_COLLECTION]

    async def ensure_indexes(self) -> None:
        logger.info("Ensuring knowledge collection indexes for %s", self.collection.name)
        await self.collection.create_index([("document_id", ASCENDING)])
        await self.collection.create_index([("chunk_index", ASCENDING)])
        await self.collection.create_index([("filename", ASCENDING)])

        try:
            index_names = list(self.collection.index_information())
            if MONGODB_KNOWLEDGE_INDEX_NAME not in index_names:
                logger.info(
                    "Creating MongoDB Atlas vector search index '%s' on field 'embedding'",
                    MONGODB_KNOWLEDGE_INDEX_NAME,
                )
                await self.collection.create_search_index(
                    model="vector-2",
                    definition={
                        "fields": [
                            {
                                "type": "vector",
                                "path": "embedding",
                                "numDimensions": 1536,
                                "similarity": "cosine",
                            }
                        ]
                    },
                    name=MONGODB_KNOWLEDGE_INDEX_NAME,
                )
            else:
                logger.info("Vector search index '%s' already exists.", MONGODB_KNOWLEDGE_INDEX_NAME)
        except Exception as exc:
            logger.warning(
                "Could not create Atlas vector search index '%s' automatically. Ensure it exists in MongoDB Atlas: %s",
                MONGODB_KNOWLEDGE_INDEX_NAME,
                exc,
                exc_info=True,
            )

    async def insert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            logger.warning("No chunks supplied for insertion.")
            return
        logger.info("Inserting %d chunks into knowledge collection %s", len(chunks), self.collection.name)
        await self.collection.insert_many(chunks)

    async def find_similar_chunks(
        self,
        query_embedding: list[float],
        limit: int = 4,
        num_candidates: int = 50,
    ) -> list[dict[str, Any]]:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": MONGODB_KNOWLEDGE_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": num_candidates,
                    "limit": limit,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "document_id": 1,
                    "filename": 1,
                    "chunk_index": 1,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        logger.info(
            "Running vector search on collection %s with limit=%s, numCandidates=%s",
            self.collection.name,
            limit,
            num_candidates,
        )

        try:
            cursor = await self.collection.aggregate(pipeline)
            results: list[dict[str, Any]] = []
            async for doc in cursor:
                results.append(doc)
        except Exception as exc:
            logger.warning(
                "Mongo vector search failed for collection %s. Check that the '%s' index exists and matches the embedding dimensions: %s",
                self.collection.name,
                MONGODB_KNOWLEDGE_INDEX_NAME,
                exc,
                exc_info=True,
            )
            raise

        logger.info("Vector search returned %d results.", len(results))
        return results

    async def ping(self) -> None:
        await self.client.admin.command("ping")
