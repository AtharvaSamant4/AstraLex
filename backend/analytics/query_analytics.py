"""
analytics/query_analytics.py — Query logging and analytical queries.

Every chat request is logged with query text, sources, latency, tier etc.
Admin analytics endpoints consume these logs for diagnostics.
"""

from __future__ import annotations

import logging
from database import crud
from database.connection import get_cursor

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Logging helper (called from chat_service after every request)
# ═══════════════════════════════════════════════════════════════════════════

def log_chat_query(
    user_id: int,
    query: str,
    session_id: str | None = None,
    rewritten_query: str | None = None,
    retrieved_sources: list[str] | None = None,
    latency_ms: float | None = None,
    tier: str | None = None,
    complexity: str | None = None,
    retrieval_chunks_count: int = 0,
) -> int:
    """Convenience wrapper around crud.log_query."""
    return crud.log_query(
        user_id=user_id,
        query=query,
        session_id=session_id,
        rewritten_query=rewritten_query,
        retrieved_sources=retrieved_sources,
        latency_ms=latency_ms,
        tier=tier,
        complexity=complexity,
        retrieval_chunks_count=retrieval_chunks_count,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Analytical queries
# ═══════════════════════════════════════════════════════════════════════════

def top_queries(limit: int = 20) -> list[dict]:
    """Return the most frequently asked queries."""
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT query,
                   COUNT(*) AS count,
                   AVG(latency_ms)::real AS avg_latency_ms,
                   MODE() WITHIN GROUP (ORDER BY tier) AS common_tier
            FROM query_logs
            GROUP BY query
            ORDER BY count DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def failure_queries(limit: int = 20) -> list[dict]:
    """
    Return queries that likely failed — no sources retrieved or
    negative feedback.
    """
    with get_cursor(commit=False) as cur:
        cur.execute(
            """
            SELECT ql.query,
                   ql.created_at,
                   ql.latency_ms,
                   ql.tier,
                   ql.retrieval_chunks_count,
                   COALESCE(array_length(ql.retrieved_sources, 1), 0) AS source_count
            FROM query_logs ql
            WHERE COALESCE(array_length(ql.retrieved_sources, 1), 0) = 0
               OR ql.response_quality = -1
            ORDER BY ql.created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def retrieval_performance() -> dict:
    """Aggregate retrieval performance metrics."""
    with get_cursor(commit=False) as cur:
        cur.execute("SELECT COUNT(*) AS total FROM query_logs")
        total = cur.fetchone()["total"]

        cur.execute(
            """
            SELECT COUNT(*) AS cnt FROM query_logs
            WHERE COALESCE(array_length(retrieved_sources, 1), 0) > 0
            """
        )
        with_sources = cur.fetchone()["cnt"]

        cur.execute(
            "SELECT AVG(retrieval_chunks_count)::real AS avg_chunks FROM query_logs"
        )
        avg_chunks = cur.fetchone()["avg_chunks"]

        cur.execute(
            """
            SELECT tier, COUNT(*) AS cnt
            FROM query_logs
            WHERE tier IS NOT NULL
            GROUP BY tier
            ORDER BY cnt DESC
            """
        )
        tier_dist = {r["tier"]: r["cnt"] for r in cur.fetchall()}

        cur.execute(
            """
            SELECT complexity, COUNT(*) AS cnt
            FROM query_logs
            WHERE complexity IS NOT NULL
            GROUP BY complexity
            ORDER BY cnt DESC
            """
        )
        complexity_dist = {r["complexity"]: r["cnt"] for r in cur.fetchall()}

        return {
            "total_queries": total,
            "queries_with_sources": with_sources,
            "retrieval_success_rate": (
                round(with_sources / total, 4) if total else 0.0
            ),
            "avg_retrieval_chunks": avg_chunks or 0.0,
            "tier_distribution": tier_dist,
            "complexity_distribution": complexity_dist,
        }
