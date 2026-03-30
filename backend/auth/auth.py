"""
auth/auth.py — Authentication helpers (bcrypt + JWT).

Provides:
  • ``hash_password`` / ``verify_password`` — bcrypt
  • ``create_access_token`` / ``decode_access_token`` — HS256 JWT
  • ``get_current_user`` — FastAPI dependency for protected routes
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from database.crud import get_user_by_id

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
_JWT_SECRET = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "24"))

_bearer_scheme = HTTPBearer()


# ═══════════════════════════════════════════════════════════════════════════
# Password hashing
# ═══════════════════════════════════════════════════════════════════════════

def hash_password(plain: str) -> str:
    """Return a bcrypt hash string for *plain* password."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Check *plain* against *hashed* (bcrypt)."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ═══════════════════════════════════════════════════════════════════════════
# JWT
# ═══════════════════════════════════════════════════════════════════════════

def create_access_token(user_id: int, email: str) -> str:
    """Create a signed JWT containing ``sub`` (user_id as string) and ``email``."""
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=_JWT_EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and verify a JWT. Raises ``HTTPException`` on failure."""
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI dependency
# ═══════════════════════════════════════════════════════════════════════════

async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer_scheme)],
) -> dict:
    """
    Dependency that extracts & validates the JWT from the
    ``Authorization: Bearer <token>`` header and returns the user dict.
    """
    payload = decode_access_token(credentials.credentials)
    raw_sub = payload.get("sub")
    if raw_sub is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    try:
        user_id = int(raw_sub)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user
