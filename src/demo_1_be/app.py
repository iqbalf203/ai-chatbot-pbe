import os
import json
import uuid
import logging
import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import configure_logging

from fastapi import (
    Depends,
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
from pymongo import AsyncMongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv

from app.api.routes.auth import router as auth_router
from app.api.routes.knowledge import router as knowledge_router
from app.core.config import MONGODB_DATABASE, MONGODB_URL
from app.core.security import get_current_user
from app.core.telemetry import AppMetrics, REQUEST_ID, TENANT_ID
from app.repositories.knowledge_repository import KnowledgeRepository
from app.services.embedding_service import EmbeddingService
from app.services.knowledge_retrieval import KnowledgeRetrievalService


# ============================================================
# Configuration
# ============================================================

load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

OPENAI_STREAM = (
    os.getenv("OPENAI_STREAM", "false").lower() == "true"
)

MONGODB_URL = os.getenv("MONGODB_URL")

MONGODB_DATABASE = os.getenv(
    "MONGODB_DATABASE",
    "ai_chatbot"
)

EMBEDDING_BASE_URL = (
    os.getenv("EMBEDDING_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
)

EMBEDDING_API_KEY = (
    os.getenv("EMBEDDING_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

EMBEDDING_MODEL = (
    os.getenv("EMBEDDING_MODEL")
    or os.getenv("OPENAI_MODEL")
)


# ============================================================
# Logging
# ============================================================

logger = configure_logging()


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are a helpful, accurate, and conversational AI assistant.

Follow these guidelines:

1. Answer the user's question directly.
2. Use Markdown to format your responses.
3. Use headings when they improve readability.
4. Use bullet points or numbered lists when appropriate.
5. Use Markdown code blocks when providing code.
6. Keep responses clear and reasonably concise.
7. Do not invent information.
8. If you are unsure about something, say so clearly.
9. Do not repeat the user's question unnecessarily.
10. Maintain context from previous messages.
"""


# ============================================================
# Global clients
# ============================================================

mongo_client = None
db = None
llm_client = None
knowledge_retrieval_service = None


# ============================================================
# Utility functions
# ============================================================

def generate_id() -> str:
    return str(uuid.uuid4())


def timestamp() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_string() -> str:
    return timestamp().isoformat()


def serialize_datetime(value):
    if isinstance(value, datetime):
        return value.isoformat()

    return value


def serialize_conversation(document):
    if not document:
        return None

    return {
        "id": document["id"],
        "title": document["title"],
        "created_at": serialize_datetime(
            document["created_at"]
        ),
        "updated_at": serialize_datetime(
            document["updated_at"]
        ),
    }


def serialize_message(document):
    if not document:
        return None

    return {
        "id": document["id"],
        "conversation_id": document["conversation_id"],
        "user_id": document.get("user_id"),
        "role": document["role"],
        "content": document["content"],
        "format": document.get(
            "format",
            "markdown"
        ),
        "created_at": serialize_datetime(
            document["created_at"]
        ),
    }


# ============================================================
# Application lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global mongo_client
    global db
    global llm_client
    global knowledge_retrieval_service

    # --------------------------------------------------------
    # MongoDB
    # --------------------------------------------------------

    if not MONGODB_URL:
        raise RuntimeError(
            "MONGODB_URL is not configured."
        )

    mongo_client = AsyncMongoClient(
        MONGODB_URL
    )

    db = mongo_client[MONGODB_DATABASE]

    # Test connection
    await mongo_client.admin.command("ping")

    logger.info(
        "Connected to MongoDB database: %s",
        MONGODB_DATABASE
    )

    # --------------------------------------------------------
    # MongoDB indexes
    # --------------------------------------------------------

    await db.users.create_index(
        [
            ("email", ASCENDING),
        ],
        unique=True,
    )

    await db.conversations.create_index(
        [
            ("user_id", ASCENDING),
            ("updated_at", DESCENDING),
        ]
    )

    await db.messages.create_index(
        [
            ("conversation_id", ASCENDING),
            ("user_id", ASCENDING),
            ("created_at", ASCENDING),
        ]
    )

    # --------------------------------------------------------
    # LLM client
    # --------------------------------------------------------

    llm_client = AsyncOpenAI(
        api_key=OPENAI_API_KEY or "dummy_api_key",
        base_url=OPENAI_BASE_URL
    )

    # --------------------------------------------------------
    # Knowledge retrieval
    # --------------------------------------------------------

    try:

        knowledge_repository = KnowledgeRepository(
            mongo_client,
            MONGODB_DATABASE
        )

        embedding_service = EmbeddingService(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
            model=EMBEDDING_MODEL,
        )

        knowledge_retrieval_service = (
            KnowledgeRetrievalService(
                repository=knowledge_repository,
                embedding_service=embedding_service,
            )
        )

    except Exception as exc:

        logger.warning(
            "Knowledge retrieval service initialization failed: %s",
            exc
        )

        knowledge_retrieval_service = None

    logger.info(
        "LLM configured: model=%s base_url=%s stream=%s",
        OPENAI_MODEL,
        OPENAI_BASE_URL,
        OPENAI_STREAM
    )

    yield

    # --------------------------------------------------------
    # Shutdown
    # --------------------------------------------------------

    if mongo_client:
        await mongo_client.close()

    logger.info(
        "Application shutdown complete"
    )


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="AI Chatbot API",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(auth_router)
app.include_router(knowledge_router)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic request models
# ============================================================

class CreateConversationRequest(BaseModel):

    title: str | None = None


class UpdateConversationRequest(BaseModel):

    title: str


class UserMessageRequest(BaseModel):

    content: str
    message_id: str | None = None


class StopRequest(BaseModel):

    message_id: str | None = None


# ============================================================
# WebSocket event helper
# ============================================================

async def send_event(
    websocket: WebSocket,
    event_type: str,
    message_id: str,
    **data
):

    event = {
        "type": event_type,
        "message_id": message_id,
        "timestamp": timestamp_string(),
        **data
    }

    await websocket.send_json(event)


# ============================================================
# Health check
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "online",
        "service": "AI Chatbot API",
        "model": OPENAI_MODEL,
        "stream": OPENAI_STREAM,
        "database": MONGODB_DATABASE,
        "websocket": "/ws/chat"
    }


@app.get("/health")
async def health():

    try:

        await mongo_client.admin.command("ping")

        return {
            "status": "healthy",
            "mongodb": "connected",
            "llm": "configured"
        }

    except Exception as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )


# ============================================================
# CREATE CONVERSATION
# ============================================================

@app.post(
    "/api/conversations",
    status_code=status.HTTP_201_CREATED
)
async def create_conversation(
    request: CreateConversationRequest,
    current_user: dict = Depends(get_current_user),
):

    conversation_id = generate_id()
    now = timestamp()

    title = (
        request.title.strip()
        if request.title
        else "New Chat"
    )

    document = {
        "id": conversation_id,
        "user_id": current_user["sub"],
        "title": title,
        "created_at": now,
        "updated_at": now,
    }

    await db.conversations.insert_one(
        document
    )

    logger.info(
        "Conversation created: %s",
        conversation_id
    )

    return serialize_conversation(
        document
    )


# ============================================================
# GET ALL CONVERSATIONS
# ============================================================

@app.get("/api/conversations")
async def get_conversations(
    current_user: dict = Depends(get_current_user)
):

    cursor = (
        db.conversations
        .find(
            {
                "user_id": current_user["sub"]
            }
        )
        .sort(
            "updated_at",
            DESCENDING
        )
    )

    conversations = []

    async for document in cursor:

        conversations.append(
            serialize_conversation(
                document
            )
        )

    return {
        "conversations": conversations
    }


# ============================================================
# GET SINGLE CONVERSATION
# ============================================================

@app.get(
    "/api/conversations/{conversation_id}"
)
async def get_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):

    conversation = await (
        db.conversations.find_one(
            {
                "id": conversation_id,
                "user_id": current_user["sub"],
            }
        )
    )

    if not conversation:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    return serialize_conversation(
        conversation
    )


# ============================================================
# UPDATE CONVERSATION TITLE
# ============================================================

@app.patch(
    "/api/conversations/{conversation_id}"
)
async def update_conversation(
    conversation_id: str,
    request: UpdateConversationRequest,
    current_user: dict = Depends(get_current_user),
):

    title = request.title.strip()

    if not title:

        raise HTTPException(
            status_code=400,
            detail="Title cannot be empty."
        )

    result = await (
        db.conversations.update_one(
            {
                "id": conversation_id,
                "user_id": current_user["sub"],
            },
            {
                "$set": {
                    "title": title,
                    "updated_at": timestamp()
                }
            }
        )
    )

    if result.matched_count == 0:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    conversation = await (
        db.conversations.find_one(
            {
                "id": conversation_id
            }
        )
    )

    return serialize_conversation(
        conversation
    )


# ============================================================
# DELETE CONVERSATION
# ============================================================

@app.delete(
    "/api/conversations/{conversation_id}"
)
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):

    result = await (
        db.conversations.delete_one(
            {
                "id": conversation_id,
                "user_id": current_user["sub"],
            }
        )
    )

    if result.deleted_count == 0:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    # Delete all messages belonging
    # to this conversation

    await db.messages.delete_many(
        {
            "conversation_id": conversation_id,
            "user_id": current_user["sub"],
        }
    )

    logger.info(
        "Conversation deleted: %s",
        conversation_id
    )

    return {
        "success": True,
        "conversation_id": conversation_id
    }


# ============================================================
# GET CONVERSATION MESSAGES
# ============================================================

@app.get(
    "/api/conversations/{conversation_id}/messages"
)
async def get_messages(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
):

    # --------------------------------------------------------
    # Verify conversation belongs to current user
    # --------------------------------------------------------

    conversation = await (
        db.conversations.find_one(
            {
                "id": conversation_id,
                "user_id": current_user["sub"],
            }
        )
    )

    if not conversation:

        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    # --------------------------------------------------------
    # Fetch messages
    # --------------------------------------------------------

    cursor = (
        db.messages
        .find(
            {
                "conversation_id": conversation_id,
                "user_id": current_user["sub"],
            }
        )
        .sort(
            "created_at",
            ASCENDING
        )
    )

    messages = []

    async for document in cursor:

        messages.append(
            serialize_message(
                document
            )
        )

    return {
        "conversation_id": conversation_id,
        "messages": messages
    }


# ============================================================
# ACTIVE STREAMS
# ============================================================

ACTIVE_STREAMS: dict[
    str,
    dict[str, Any]
] = {}


# ============================================================
# METRICS
# ============================================================

@app.get("/metrics")
async def metrics_endpoint():

    return AppMetrics.snapshot()


# ============================================================
# WEBSOCKET CHAT
# ============================================================

@app.websocket("/ws/chat")
async def websocket_endpoint(
    websocket: WebSocket
):

    await websocket.accept()

    request_id = generate_id()

    REQUEST_ID.set(request_id)
    TENANT_ID.set("default")

    # --------------------------------------------------------
    # Conversation ID
    # --------------------------------------------------------

    conversation_id = (
        websocket.query_params
        .get("conversation_id")
    )

    if not conversation_id:

        await send_event(
            websocket,
            "error",
            "",
            code="MISSING_CONVERSATION_ID",
            message=(
                "conversation_id query parameter "
                "is required."
            )
        )

        await websocket.close(
            code=1008
        )

        return

    # --------------------------------------------------------
    # Verify conversation
    # --------------------------------------------------------

    conversation = await (
        db.conversations.find_one(
            {
                "id": conversation_id
            }
        )
    )

    if not conversation:

        await send_event(
            websocket,
            "error",
            "",
            code="CONVERSATION_NOT_FOUND",
            message="Conversation not found."
        )

        await websocket.close(
            code=1008
        )

        return

    # --------------------------------------------------------
    # User ID comes from the conversation
    #
    # WebSocket authentication is intentionally NOT used.
    # --------------------------------------------------------

    user_id = conversation.get("user_id")

    logger.info(
        "WebSocket connected for conversation=%s user=%s",
        conversation_id,
        user_id
    )

    try:

        while True:

            # =================================================
            # Receive message
            # =================================================

            raw_data = await (
                websocket.receive_text()
            )

            # =================================================
            # Parse JSON
            # =================================================

            try:

                payload = json.loads(
                    raw_data
                )

            except json.JSONDecodeError:

                await send_event(
                    websocket,
                    "error",
                    "",
                    code="INVALID_JSON",
                    message="Invalid JSON payload."
                )

                continue

            # =================================================
            # Stop generation
            # =================================================

            if payload.get(
                "type"
            ) == "stop_generation":

                message_id = payload.get(
                    "message_id"
                )

                if (
                    message_id
                    and message_id in ACTIVE_STREAMS
                ):

                    task = (
                        ACTIVE_STREAMS[
                            message_id
                        ].get("task")
                    )

                    ACTIVE_STREAMS[
                        message_id
                    ]["cancelled"] = True

                    if (
                        task
                        and not task.done()
                    ):

                        task.cancel()

                    await send_event(
                        websocket,
                        "generation_stopped",
                        message_id,
                        status="stopped",
                    )

                continue

            # =================================================
            # Validate message type
            # =================================================

            if payload.get(
                "type"
            ) != "user_message":

                await send_event(
                    websocket,
                    "error",
                    "",
                    code="INVALID_MESSAGE_TYPE",
                    message=(
                        "Expected message type "
                        "'user_message'."
                    )
                )

                continue

            # =================================================
            # Extract user content
            # =================================================

            user_content = (
                payload
                .get("content", "")
                .strip()
            )

            if not user_content:

                await send_event(
                    websocket,
                    "error",
                    "",
                    code="EMPTY_MESSAGE",
                    message=(
                        "Message content cannot be empty."
                    )
                )

                continue

            # =================================================
            # Message ID
            # =================================================

            user_message_id = (
                payload.get(
                    "message_id"
                )
                or generate_id()
            )

            # =================================================
            # Persist USER message
            # =================================================

            user_message = {
                "id": user_message_id,

                "conversation_id":
                    conversation_id,

                # NEW
                "user_id":
                    user_id,

                "role": "user",

                "content":
                    user_content,

                "format":
                    "markdown",

                "created_at":
                    timestamp()
            }

            await db.messages.insert_one(
                user_message
            )

            logger.info(
                "User message stored: "
                "message=%s conversation=%s user=%s",
                user_message_id,
                conversation_id,
                user_id
            )

            # =================================================
            # Update conversation
            # =================================================

            if conversation["title"] == "New Chat":

                generated_title = (
                    user_content[:50]
                )

                if len(user_content) > 50:

                    generated_title += "..."

                await db.conversations.update_one(
                    {
                        "id":
                            conversation_id
                    },
                    {
                        "$set": {
                            "title":
                                generated_title,

                            "updated_at":
                                timestamp()
                        }
                    }
                )

                conversation["title"] = (
                    generated_title
                )

            else:

                await db.conversations.update_one(
                    {
                        "id":
                            conversation_id
                    },
                    {
                        "$set": {
                            "updated_at":
                                timestamp()
                        }
                    }
                )

            # =================================================
            # Load conversation history
            # =================================================

            cursor = (
                db.messages
                .find(
                    {
                        "conversation_id":
                            conversation_id
                    }
                )
                .sort(
                    "created_at",
                    ASCENDING
                )
            )

            history = [
                {
                    "role": "system",
                    "content":
                        SYSTEM_PROMPT
                }
            ]

            # =================================================
            # Knowledge retrieval
            # =================================================

            knowledge_context = ""

            if knowledge_retrieval_service:

                try:

                    logger.info(
                        "Starting knowledge retrieval "
                        "for chat question: %s",
                        user_content[:200]
                    )

                    knowledge_context = (
                        await knowledge_retrieval_service.retrieve(
                            user_content
                        )
                    )

                    if knowledge_context:

                        logger.info(
                            "Knowledge retrieval returned "
                            "%d chars of context for question: %s",
                            len(knowledge_context),
                            user_content[:200]
                        )

                    else:

                        logger.info(
                            "Knowledge retrieval returned "
                            "no context for question: %s",
                            user_content[:200]
                        )

                except Exception as exc:

                    logger.warning(
                        "Knowledge retrieval failed "
                        "for chat query: %s",
                        exc,
                        exc_info=True
                    )

                    knowledge_context = ""

            # =================================================
            # Add knowledge context
            # =================================================

            if knowledge_context:

                history.append(
                    {
                        "role": "system",
                        "content": (
                            "Use the following internal "
                            "knowledge as context when relevant. "
                            "If the answer is not in the context, "
                            "say that clearly and answer from "
                            "general knowledge only.\n\n"
                            f"{knowledge_context}"
                        )
                    }
                )

            # =================================================
            # Add conversation history
            # =================================================

            async for message in cursor:

                history.append(
                    {
                        "role":
                            message["role"],

                        "content":
                            message["content"]
                    }
                )

            # =================================================
            # Create assistant message ID
            # =================================================

            assistant_message_id = (
                generate_id()
            )

            ACTIVE_STREAMS[
                assistant_message_id
            ] = {
                "cancelled": False,
                "task": None
            }

            # =================================================
            # Notify UI
            # =================================================

            await send_event(
                websocket,
                "message_start",
                assistant_message_id,
                role="assistant",
                request_id=request_id,
            )

            assistant_response = ""

            # =================================================
            # LLM
            # =================================================

            try:

                async def run_llm_call() -> Any:

                    logger.info(
                        "Calling model=%s stream=%s",
                        OPENAI_MODEL,
                        OPENAI_STREAM
                    )

                    return await (
                        llm_client
                        .chat
                        .completions
                        .create(
                            model=OPENAI_MODEL,
                            messages=history,
                            stream=OPENAI_STREAM
                        )
                    )

                task = asyncio.create_task(
                    run_llm_call()
                )

                ACTIVE_STREAMS[
                    assistant_message_id
                ]["task"] = task

                response = await task

                # =================================================
                # STREAMING
                # =================================================

                if OPENAI_STREAM:

                    async for chunk in response:

                        if (
                            ACTIVE_STREAMS
                            .get(
                                assistant_message_id,
                                {}
                            )
                            .get("cancelled")
                        ):

                            await send_event(
                                websocket,
                                "generation_stopped",
                                assistant_message_id,
                                status="stopped",
                            )

                            break

                        if not chunk.choices:

                            continue

                        content_chunk = (
                            chunk
                            .choices[0]
                            .delta
                            .content
                        )

                        if not content_chunk:

                            continue

                        assistant_response += (
                            content_chunk
                        )

                        await send_event(
                            websocket,
                            "message_delta",
                            assistant_message_id,
                            delta=content_chunk,
                            format="markdown",
                            request_id=request_id,
                        )

                    # ------------------------------------------------
                    # Generation cancelled
                    # ------------------------------------------------

                    if (
                        ACTIVE_STREAMS
                        .get(
                            assistant_message_id,
                            {}
                        )
                        .get("cancelled")
                    ):

                        ACTIVE_STREAMS.pop(
                            assistant_message_id,
                            None
                        )

                        await send_event(
                            websocket,
                            "message_end",
                            assistant_message_id,
                            status="stopped",
                            request_id=request_id,
                        )

                        continue

                    # ------------------------------------------------
                    # Persist assistant response
                    # ------------------------------------------------

                    assistant_message = {

                        "id":
                            assistant_message_id,

                        "conversation_id":
                            conversation_id,

                        # NEW
                        "user_id":
                            user_id,

                        "role":
                            "assistant",

                        "content":
                            assistant_response,

                        "format":
                            "markdown",

                        "created_at":
                            timestamp()
                    }

                    await db.messages.insert_one(
                        assistant_message
                    )

                    # ------------------------------------------------
                    # Update conversation
                    # ------------------------------------------------

                    await db.conversations.update_one(
                        {
                            "id":
                                conversation_id
                        },
                        {
                            "$set": {
                                "updated_at":
                                    timestamp()
                            }
                        }
                    )

                    # ------------------------------------------------
                    # End streaming
                    # ------------------------------------------------

                    await send_event(
                        websocket,
                        "message_end",
                        assistant_message_id
                    )

                # =================================================
                # NON STREAMING
                # =================================================

                else:

                    if not response.choices:

                        raise ValueError(
                            "Model returned no choices."
                        )

                    assistant_response = (
                        response
                        .choices[0]
                        .message
                        .content
                        or ""
                    )

                    # ------------------------------------------------
                    # Persist assistant response
                    # ------------------------------------------------

                    assistant_message = {

                        "id":
                            assistant_message_id,

                        "conversation_id":
                            conversation_id,

                        # NEW
                        "user_id":
                            user_id,

                        "role":
                            "assistant",

                        "content":
                            assistant_response,

                        "format":
                            "markdown",

                        "created_at":
                            timestamp()
                    }

                    await db.messages.insert_one(
                        assistant_message
                    )

                    # ------------------------------------------------
                    # Update conversation
                    # ------------------------------------------------

                    await db.conversations.update_one(
                        {
                            "id":
                                conversation_id
                        },
                        {
                            "$set": {
                                "updated_at":
                                    timestamp()
                            }
                        }
                    )

                    # ------------------------------------------------
                    # Send complete response
                    # ------------------------------------------------

                    await send_event(
                        websocket,
                        "message_complete",
                        assistant_message_id,
                        content=assistant_response,
                        format="markdown"
                    )

                    # ------------------------------------------------
                    # Universal end event
                    # ------------------------------------------------

                    await send_event(
                        websocket,
                        "message_end",
                        assistant_message_id
                    )

            except Exception as e:

                logger.exception(
                    "LLM error"
                )

                await send_event(
                    websocket,
                    "error",
                    assistant_message_id,
                    code="MODEL_ERROR",
                    message=str(e)
                )

            finally:

                ACTIVE_STREAMS.pop(
                    assistant_message_id,
                    None
                )

    except WebSocketDisconnect:

        logger.info(
            "WebSocket disconnected: %s",
            conversation_id
        )

    except Exception as e:

        logger.exception(
            "WebSocket error"
        )