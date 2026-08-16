import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "ai_chatbot")
MONGODB_KNOWLEDGE_COLLECTION = os.getenv(
    "MONGODB_KNOWLEDGE_COLLECTION",
    "knowledge_documents",
)

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openrouter")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or os.getenv("OPENAI_MODEL")

MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", str(10 * 1024 * 1024)))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
KNOWLEDGE_SEARCH_LIMIT = int(os.getenv("KNOWLEDGE_SEARCH_LIMIT", "4"))
KNOWLEDGE_SEARCH_CANDIDATES = int(os.getenv("KNOWLEDGE_SEARCH_CANDIDATES", "50"))
MONGODB_KNOWLEDGE_INDEX_NAME = os.getenv("MONGODB_KNOWLEDGE_INDEX_NAME", "knowledge_embedding_vector_index")
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".xlsx"}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]
