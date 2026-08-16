import unittest
from unittest.mock import AsyncMock, MagicMock

from app.repositories.knowledge_repository import KnowledgeRepository


class KnowledgeRepositoryIndexTests(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_indexes_creates_vector_search_index(self):
        collection = MagicMock()
        collection.create_index = AsyncMock()
        collection.create_search_index = AsyncMock(return_value="vector-index-created")

        mongo_client = MagicMock()
        mongo_client.__getitem__.return_value.__getitem__.return_value = collection

        repo = KnowledgeRepository(mongo_client, db_name="ai_chatbot")

        await repo.ensure_indexes()

        collection.create_search_index.assert_awaited_once()
        self.assertEqual(collection.create_search_index.await_args.kwargs["definition"]["fields"][0]["path"], "embedding")


if __name__ == "__main__":
    unittest.main()
