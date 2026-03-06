"""
database/schema.py — DDL for all tables.

All statements are idempotent (IF NOT EXISTS).
"""

from __future__ import annotations

import logging

from database.connection import get_cursor

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """\
-- ── Users ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              SERIAL PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Chat sessions ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_sessions (
    id              TEXT PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Messages ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_messages (
    id              SERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    sources         TEXT[] DEFAULT '{}',
    rewritten_query TEXT,
    complexity      TEXT,
    tier            TEXT,
    timings         JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Feedback ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_feedback (
    id              SERIAL PRIMARY KEY,
    message_id      INTEGER NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
    rating          SMALLINT CHECK (rating IN (-1, 1)),
    comment         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Documents (user uploads) ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    file_type       TEXT NOT NULL,
    title           TEXT,
    total_chunks    INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'processing'
                        CHECK (status IN ('processing', 'ready', 'failed')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Document chunks ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_chunks (
    id              SERIAL PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding_ref   TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Query logs ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_logs (
    id              SERIAL PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id      TEXT REFERENCES chat_sessions(id) ON DELETE SET NULL,
    query           TEXT NOT NULL,
    rewritten_query TEXT,
    retrieved_sources TEXT[] DEFAULT '{}',
    latency_ms      REAL,
    tier            TEXT,
    complexity      TEXT,
    retrieval_chunks_count INTEGER DEFAULT 0,
    response_quality SMALLINT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Indexes ───────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
    ON chat_sessions(user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_chat_feedback_message
    ON chat_feedback(message_id);

CREATE INDEX IF NOT EXISTS idx_users_email
    ON users(email);

CREATE INDEX IF NOT EXISTS idx_documents_user
    ON documents(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_document_chunks_doc
    ON document_chunks(document_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_query_logs_user
    ON query_logs(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_query_logs_session
    ON query_logs(session_id);

CREATE INDEX IF NOT EXISTS idx_query_logs_created
    ON query_logs(created_at DESC);
"""


def init_schema() -> None:
    """Create all tables (idempotent)."""
    with get_cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    logger.info("Database schema initialised (all tables)")
