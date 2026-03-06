"""
database.py — PostgreSQL (Neon) persistence for the Legal RAG Chatbot.

Stores conversation sessions and individual messages so that:
  • Users can resume previous conversations
  • Analytics / monitoring dashboards can track usage
  • Feedback ratings can improve the system over time

Tables
------
sessions     – one row per conversation session
messages     – one row per user/assistant exchange within a session
feedback     – optional per-message thumbs up/down + text

All DDL is idempotent (IF NOT EXISTS).
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Connection helpers ─────────────────────────────────────────────────────

_DATABASE_URL: str = os.getenv("DATABASE_URL", "")


def _get_conn():
    """Return a new psycopg2 connection from DATABASE_URL."""
    if not _DATABASE_URL:
        raise EnvironmentError("DATABASE_URL is not set in .env")
    return psycopg2.connect(_DATABASE_URL)


@contextmanager
def get_cursor(commit: bool = True):
    """Context manager yielding a dict-cursor; auto-commits on success."""
    conn = _get_conn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
            if commit:
                conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema (idempotent) ───────────────────────────────────────────────────

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    title       TEXT,
    metadata    JSONB DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS messages (
    id          SERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    sources     TEXT[] DEFAULT '{}',
    rewritten_query TEXT,
    complexity  TEXT,
    tier        TEXT,
    timings     JSONB DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feedback (
    id          SERIAL PRIMARY KEY,
    message_id  INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    rating      SMALLINT CHECK (rating IN (-1, 1)),
    comment     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_feedback_message
    ON feedback(message_id);
"""


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    logger.info("Database schema initialised (Neon PostgreSQL)")


# ── Session CRUD ───────────────────────────────────────────────────────────

def create_session(title: str | None = None) -> str:
    """Create a new conversation session; returns the session id."""
    session_id = uuid.uuid4().hex[:16]
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (id, title) VALUES (%s, %s)",
            (session_id, title),
        )
    logger.info("Created session %s", session_id)
    return session_id


def list_sessions(limit: int = 20) -> list[dict]:
    """Return the most recent sessions."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            "SELECT id, title, created_at, updated_at "
            "FROM sessions ORDER BY updated_at DESC LIMIT %s",
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_session(session_id: str) -> dict | None:
    """Fetch a single session by id."""
    with get_cursor(commit=False) as cur:
        cur.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def delete_session(session_id: str) -> bool:
    """Delete a session and all its messages/feedback. Returns True if found."""
    with get_cursor() as cur:
        cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
        return cur.rowcount > 0


def _touch_session(cur, session_id: str) -> None:
    """Update the session's updated_at timestamp."""
    cur.execute(
        "UPDATE sessions SET updated_at = NOW() WHERE id = %s",
        (session_id,),
    )


# ── Message CRUD ───────────────────────────────────────────────────────────

def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[str] | None = None,
    rewritten_query: str | None = None,
    complexity: str | None = None,
    tier: str | None = None,
    timings: dict | None = None,
) -> int:
    """Insert a message and return its id."""
    import json

    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO messages
                (session_id, role, content, sources, rewritten_query,
                 complexity, tier, timings)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                session_id, role, content,
                sources or [],
                rewritten_query,
                complexity,
                tier,
                json.dumps(timings or {}),
            ),
        )
        msg_id = cur.fetchone()["id"]
        _touch_session(cur, session_id)
    return msg_id


def get_messages(session_id: str) -> list[dict]:
    """Return all messages for a session, ordered chronologically."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            "SELECT * FROM messages WHERE session_id = %s ORDER BY created_at",
            (session_id,),
        )
        return [dict(r) for r in cur.fetchall()]


# ── Feedback ───────────────────────────────────────────────────────────────

def save_feedback(message_id: int, rating: int, comment: str | None = None) -> int:
    """Save feedback for a message. rating: 1 (good) or -1 (bad)."""
    with get_cursor() as cur:
        cur.execute(
            "INSERT INTO feedback (message_id, rating, comment) "
            "VALUES (%s, %s, %s) RETURNING id",
            (message_id, rating, comment),
        )
        return cur.fetchone()["id"]


# ── Analytics helpers ──────────────────────────────────────────────────────

def get_stats() -> dict:
    """Return basic usage statistics."""
    with get_cursor(commit=False) as cur:
        cur.execute("SELECT COUNT(*) AS cnt FROM sessions")
        total_sessions = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) AS cnt FROM messages")
        total_messages = cur.fetchone()["cnt"]

        cur.execute(
            "SELECT COUNT(*) AS cnt FROM messages WHERE role = 'user'"
        )
        total_questions = cur.fetchone()["cnt"]

        cur.execute(
            "SELECT AVG(rating)::float AS avg FROM feedback"
        )
        avg_rating = cur.fetchone()["avg"]

        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_questions": total_questions,
            "avg_feedback_rating": avg_rating,
        }
