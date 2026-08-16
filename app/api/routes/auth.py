from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

from app.core.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

router = APIRouter(
    prefix="/api/auth",
    tags=["auth"],
)


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    name: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    email: str
    name: str | None = None
    created_at: str | None = None


async def get_db_users():
    import importlib

    chat_app = importlib.import_module("demo_1_be.app")

    db = getattr(chat_app, "db", None)

    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Database is not available",
        )

    return db.users


@router.post(
    "/signup",
    status_code=status.HTTP_201_CREATED,
)
async def signup(payload: SignUpRequest):
    users_collection = await get_db_users()

    email = payload.email.lower().strip()

    existing = await users_collection.find_one(
        {"email": email}
    )

    if existing:
        raise HTTPException(
            status_code=409,
            detail="User already exists",
        )

    user_id = str(uuid.uuid4())

    now = datetime.now(timezone.utc)

    user = {
        "id": user_id,
        "email": email,
        "name": (
            payload.name.strip()
            if payload.name
            else email.split("@")[0]
        ),
        "password_hash": hash_password(
            payload.password
        ),
        "created_at": now,
        "updated_at": now,
        "is_active": True,
    }

    await users_collection.insert_one(user)

    return {
        "user": {
            "id": user_id,
            "email": email,
            "name": user["name"],
        }
    }


@router.post(
    "/login",
    response_model=TokenResponse,
)
async def login(payload: LoginRequest):
    users_collection = await get_db_users()

    email = payload.email.lower().strip()

    user = await users_collection.find_one(
        {"email": email}
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    if not verify_password(
        payload.password,
        user["password_hash"],
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    if not user.get("is_active", True):
        raise HTTPException(
            status_code=403,
            detail="User account is inactive",
        )

    token = create_access_token(
        {
            "sub": user["id"],
            "email": user["email"],
            "name": user.get(
                "name",
                email.split("@")[0],
            ),
        }
    )

    return {
        "access_token": token,
        "token_type": "bearer",
    }


@router.get(
    "/me",
    response_model=UserResponse,
)
async def me(
    current_user: dict = Depends(
        get_current_user
    ),
):
    users_collection = await get_db_users()

    user = await users_collection.find_one(
        {"id": current_user["sub"]}
    )

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found",
        )

    created_at = user.get("created_at")

    if hasattr(created_at, "isoformat"):
        created_at = created_at.isoformat()
    elif created_at is not None:
        created_at = str(created_at)

    return {
        "id": user["id"],
        "email": user["email"],
        "name": user.get("name"),
        "created_at": created_at,
    }