from __future__ import annotations

from openai import AsyncOpenAI

from app.core.config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL


class EmbeddingService:
    def __init__(
        self,
        api_key: str | None = EMBEDDING_API_KEY,
        base_url: str | None = EMBEDDING_BASE_URL,
        model: str | None = EMBEDDING_MODEL,
    ):
        self.model = model or "text-embedding-3-small"
        self.client = AsyncOpenAI(api_key=api_key or "dummy_api_key", base_url=base_url)

    async def generate_embedding(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Embedding input text cannot be empty.")

        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as exc:
            raise RuntimeError(f"Embedding generation failed: {exc}") from exc
