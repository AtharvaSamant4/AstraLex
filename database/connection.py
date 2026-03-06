"""
database/connection.py — PostgreSQL connection pool for the Legal RAG Chatbot.

Provides a shared connection helper and cursor context manager used by
all database modules.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

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
