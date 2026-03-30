"""
database/crud.py — CRUD operations for users, chat_sessions, chat_messages, chat_feedback.

Every function uses the shared ``get_cursor()`` from ``database.connection``.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from database.connection import get_cursor

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Users
# ═══════════════════════════════════════════════════════════════════════════

def create_user(email: str, password_hash: str) -> dict:
    """Insert a new user and return ``{id, email, created_at}``."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (email, password_hash)
            VALUES (%s, %s)
            RETURNING id, email, created_at
            """,
            (email, password_hash),
        )
        return dict(cur.fetchone())


def get_user_by_email(email: str) -> Optional[dict]:
    """Fetch a user row by e-mail (includes password_hash)."""
    with get_cursor(commit=False) as cur:
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    """Fetch a user row by primary key."""
    with get_cursor(commit=False) as cur:
        cur.execute("SELECT id, email, created_at FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        return dict(row) if row else None


# ═══════════════════════════════════════════════════════════════════════════
# Chat Sessions
# ═══════════════════════════════════════════════════════════════════════════

def create_session(user_id: int, title: Optional[str] = None) -> dict:
    """Create a new chat session for a user."""
    session_id = uuid.uuid4().hex[:16]
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_sessions (id, user_id, title)
            VALUES (%s, %s, %s)
            RETURNING id, user_id, title, created_at, updated_at
            """,
            (session_id, user_id, title),
        )
        return dict(cur.fetchone())


def list_sessions(user_id: int, limit: int = 20) -> list[dict]:
    """Return the most recent sessions for a user."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT id, user_id, title, created_at, updated_at
            FROM chat_sessions
            WHERE user_id = %s
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]


def get_session(session_id: str, user_id: int) -> Optional[dict]:
    """Fetch a single session ensuring it belongs to the given user."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            "SELECT * FROM chat_sessions WHERE id = %s AND user_id = %s",
            (session_id, user_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def delete_session(session_id: str, user_id: int) -> bool:
    """Delete a session (and its messages via CASCADE). Returns True if found."""
    with get_cursor() as cur:
        cur.execute(
            "DELETE FROM chat_sessions WHERE id = %s AND user_id = %s",
            (session_id, user_id),
        )
        return cur.rowcount > 0


def update_session_title(session_id: str, user_id: int, title: str) -> bool:
    """Update a session's title. Returns True if the row was found."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE chat_sessions SET title = %s, updated_at = NOW()
            WHERE id = %s AND user_id = %s
            """,
            (title, session_id, user_id),
        )
        return cur.rowcount > 0


def _touch_session(cur, session_id: str) -> None:
    """Bump ``updated_at`` (called inside an existing cursor context)."""
    cur.execute(
        "UPDATE chat_sessions SET updated_at = NOW() WHERE id = %s",
        (session_id,),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Chat Messages
# ═══════════════════════════════════════════════════════════════════════════

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
    """Insert a message and return its id. Also bumps ``updated_at``."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_messages
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


def update_message_content(
    message_id: int,
    content: str,
    sources: list[str] | None = None,
    rewritten_query: str | None = None,
    complexity: str | None = None,
    tier: str | None = None,
) -> None:
    """Update an existing message's content and metadata."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE chat_messages
            SET content = %s,
                sources = %s,
                rewritten_query = %s,
                complexity = %s,
                tier = %s
            WHERE id = %s
            """,
            (
                content,
                sources or [],
                rewritten_query,
                complexity,
                tier,
                message_id,
            ),
        )


def get_messages(session_id: str, limit: int | None = None) -> list[dict]:
    """Return messages for a session ordered chronologically.

    If *limit* is given, return only the last *limit* messages (useful for
    building the memory window).
    """
    with get_cursor(commit=False) as cur:
        if limit:
            cur.execute(
                """
                SELECT * FROM (
                    SELECT * FROM chat_messages
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) sub ORDER BY created_at
                """,
                (session_id, limit),
            )
        else:
            cur.execute(
                "SELECT * FROM chat_messages WHERE session_id = %s ORDER BY created_at",
                (session_id,),
            )
        return [dict(r) for r in cur.fetchall()]


def get_recent_messages(session_id: str, n: int = 10) -> list[dict]:
    """Return the last *n* messages for a session (for memory window)."""
    return get_messages(session_id, limit=n)


# ═══════════════════════════════════════════════════════════════════════════
# Feedback
# ═══════════════════════════════════════════════════════════════════════════

def save_feedback(message_id: int, rating: int, comment: str | None = None) -> int:
    """Save feedback for a message; returns feedback id."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO chat_feedback (message_id, rating, comment)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (message_id, rating, comment),
        )
        return cur.fetchone()["id"]


# ═══════════════════════════════════════════════════════════════════════════
# Analytics
# ═══════════════════════════════════════════════════════════════════════════

def get_stats(user_id: int | None = None) -> dict:
    """Return basic usage statistics, optionally scoped to a single user."""
    with get_cursor(commit=False) as cur:
        if user_id:
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM chat_sessions WHERE user_id = %s",
                (user_id,),
            )
        else:
            cur.execute("SELECT COUNT(*) AS cnt FROM chat_sessions")
        total_sessions = cur.fetchone()["cnt"]

        if user_id:
            cur.execute(
                """
                SELECT COUNT(*) AS cnt FROM chat_messages m
                JOIN chat_sessions s ON s.id = m.session_id
                WHERE s.user_id = %s
                """,
                (user_id,),
            )
        else:
            cur.execute("SELECT COUNT(*) AS cnt FROM chat_messages")
        total_messages = cur.fetchone()["cnt"]

        if user_id:
            cur.execute(
                """
                SELECT COUNT(*) AS cnt FROM chat_messages m
                JOIN chat_sessions s ON s.id = m.session_id
                WHERE s.user_id = %s AND m.role = 'user'
                """,
                (user_id,),
            )
        else:
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM chat_messages WHERE role = 'user'"
            )
        total_questions = cur.fetchone()["cnt"]

        cur.execute("SELECT AVG(rating)::float AS avg FROM chat_feedback")
        avg_rating = cur.fetchone()["avg"]

        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_questions": total_questions,
            "avg_feedback_rating": avg_rating,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Documents
# ═══════════════════════════════════════════════════════════════════════════

def create_document(
    user_id: int,
    filename: str,
    file_type: str,
    title: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Insert a document record (status='processing')."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (user_id, filename, file_type, title, session_id)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id, user_id, filename, file_type, title, total_chunks,
                      status, created_at, session_id
            """,
            (user_id, filename, file_type, title or filename, session_id),
        )
        return dict(cur.fetchone())


def update_document_status(
    doc_id: int, status: str, total_chunks: int = 0,
) -> None:
    """Mark a document as 'ready' or 'failed'."""
    with get_cursor() as cur:
        cur.execute(
            "UPDATE documents SET status = %s, total_chunks = %s WHERE id = %s",
            (status, total_chunks, doc_id),
        )


def list_documents(user_id: int) -> list[dict]:
    """List all documents for a user."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT id, user_id, filename, file_type, title, total_chunks,
                   status, created_at, session_id
            FROM documents
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_ready_documents(user_id: int) -> list[dict]:
    """List only READY documents for a user (for retrieval)."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT id, user_id, filename, file_type, title, total_chunks,
                   status, created_at, session_id
            FROM documents
            WHERE user_id = %s AND status = 'ready'
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def list_session_documents(session_id: str, user_id: int) -> list[dict]:
    """List documents attached to a specific session."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT id, user_id, filename, file_type, title, total_chunks,
                   status, created_at, session_id
            FROM documents
            WHERE session_id = %s AND user_id = %s
            ORDER BY created_at DESC
            """,
            (session_id, user_id),
        )
        return [dict(r) for r in cur.fetchall()]


def get_document(doc_id: int, user_id: int) -> Optional[dict]:
    """Fetch a document ensuring it belongs to the user."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            "SELECT * FROM documents WHERE id = %s AND user_id = %s",
            (doc_id, user_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def delete_document(doc_id: int, user_id: int) -> bool:
    """Delete a document and its chunks (CASCADE). Returns True if found."""
    with get_cursor() as cur:
        cur.execute(
            "DELETE FROM documents WHERE id = %s AND user_id = %s",
            (doc_id, user_id),
        )
        return cur.rowcount > 0


# ═══════════════════════════════════════════════════════════════════════════
# Document chunks
# ═══════════════════════════════════════════════════════════════════════════

def save_document_chunks(
    doc_id: int,
    chunks: list[dict],
) -> int:
    """Bulk-insert chunks for a document. Each dict has 'chunk_index',
    'chunk_text', and optionally 'embedding_ref'. Returns count inserted."""
    if not chunks:
        return 0
    with get_cursor() as cur:
        from psycopg2.extras import execute_values
        values = [
            (doc_id, c["chunk_index"], c["chunk_text"], c.get("embedding_ref"))
            for c in chunks
        ]
        execute_values(
            cur,
            """
            INSERT INTO document_chunks (document_id, chunk_index, chunk_text, embedding_ref)
            VALUES %s
            """,
            values,
        )
        return len(values)


def get_document_chunks(doc_id: int) -> list[dict]:
    """Return all chunks for a document."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            "SELECT * FROM document_chunks WHERE document_id = %s ORDER BY chunk_index",
            (doc_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def get_user_document_chunks(user_id: int) -> list[dict]:
    """Return ALL ready-document chunks for a user (for retrieval)."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT dc.id, dc.document_id, dc.chunk_index, dc.chunk_text,
                   d.filename, d.title
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            WHERE d.user_id = %s AND d.status = 'ready'
            ORDER BY dc.document_id, dc.chunk_index
            """,
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]


# ═══════════════════════════════════════════════════════════════════════════
# Password reset tokens
# ═══════════════════════════════════════════════════════════════════════════

def create_password_reset_token(user_id: int, token: str, expires_at) -> int:
    """Insert a reset token and return its id."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO password_reset_tokens (user_id, token, expires_at)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (user_id, token, expires_at),
        )
        return cur.fetchone()["id"]


def get_valid_reset_token(token: str) -> Optional[dict]:
    """Fetch a non-expired, unused reset token."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT * FROM password_reset_tokens
            WHERE token = %s AND used = FALSE AND expires_at > NOW()
            """,
            (token,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def mark_reset_token_used(token: str) -> None:
    """Mark a reset token as used."""
    with get_cursor() as cur:
        cur.execute(
            "UPDATE password_reset_tokens SET used = TRUE WHERE token = %s",
            (token,),
        )


def update_user_password(user_id: int, password_hash: str) -> None:
    """Update a user's password hash."""
    with get_cursor() as cur:
        cur.execute(
            "UPDATE users SET password_hash = %s WHERE id = %s",
            (password_hash, user_id),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Query logs
# ═══════════════════════════════════════════════════════════════════════════

def log_query(
    user_id: int,
    query: str,
    session_id: str | None = None,
    rewritten_query: str | None = None,
    retrieved_sources: list[str] | None = None,
    latency_ms: float | None = None,
    tier: str | None = None,
    complexity: str | None = None,
    retrieval_chunks_count: int = 0,
    response_quality: int | None = None,
) -> int:
    """Insert a query log row and return its id."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO query_logs
                (user_id, session_id, query, rewritten_query, retrieved_sources,
                 latency_ms, tier, complexity, retrieval_chunks_count,
                 response_quality)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                user_id, session_id, query, rewritten_query,
                retrieved_sources or [],
                latency_ms, tier, complexity, retrieval_chunks_count,
                response_quality,
            ),
        )
        return cur.fetchone()["id"]


def get_query_logs(
    user_id: int | None = None, limit: int = 100,
) -> list[dict]:
    """Return recent query logs, optionally scoped to a user."""
    with get_cursor(commit=False) as cur:
        if user_id:
            cur.execute(
                "SELECT * FROM query_logs WHERE user_id = %s "
                "ORDER BY created_at DESC LIMIT %s",
                (user_id, limit),
            )
        else:
            cur.execute(
                "SELECT * FROM query_logs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
        return [dict(r) for r in cur.fetchall()]
