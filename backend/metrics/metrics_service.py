"""
metrics/metrics_service.py — System monitoring metrics.

Collects operational metrics from query_logs + chat_feedback tables and
exposes them via ``get_system_metrics()``.

Metrics returned:
  • total_queries
  • average_latency_ms
  • p95_latency_ms
  • hallucination_rate  (negative feedback / total feedback)
  • retrieval_success_rate
  • queries_per_user  (avg)
  • active_users  (users who asked ≥1 query)
  • total_documents_uploaded
  • total_document_chunks
"""

from __future__ import annotations

import logging
from database.connection import get_cursor

logger = logging.getLogger(__name__)


def get_system_metrics() -> dict:
    """Return a comprehensive metrics snapshot for monitoring dashboards."""
    with get_cursor(commit=False) as cur:
        # ── Total queries ──────────────────────────────────────────────
        cur.execute("SELECT COUNT(*) AS cnt FROM query_logs")
        total_queries = cur.fetchone()["cnt"]

        # ── Average latency ────────────────────────────────────────────
        cur.execute(
            "SELECT AVG(latency_ms)::real AS avg FROM query_logs "
            "WHERE latency_ms IS NOT NULL"
        )
        avg_latency = cur.fetchone()["avg"] or 0.0

        # ── p95 latency ───────────────────────────────────────────────
        cur.execute(
            """
            SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)
                   AS p95
            FROM query_logs
            WHERE latency_ms IS NOT NULL
            """
        )
        p95_latency = cur.fetchone()["p95"] or 0.0

        # ── Hallucination rate (negative feedback / total feedback) ────
        cur.execute("SELECT COUNT(*) AS cnt FROM chat_feedback")
        total_feedback = cur.fetchone()["cnt"]

        cur.execute(
            "SELECT COUNT(*) AS cnt FROM chat_feedback WHERE rating = -1"
        )
        negative_feedback = cur.fetchone()["cnt"]

        hallucination_rate = (
            round(negative_feedback / total_feedback, 4)
            if total_feedback > 0 else 0.0
        )

        # ── Retrieval success rate ────────────────────────────────────
        cur.execute(
            """
            SELECT COUNT(*) AS cnt FROM query_logs
            WHERE COALESCE(array_length(retrieved_sources, 1), 0) > 0
            """
        )
        queries_with_sources = cur.fetchone()["cnt"]

        retrieval_success = (
            round(queries_with_sources / total_queries, 4)
            if total_queries > 0 else 0.0
        )

        # ── Queries per user ──────────────────────────────────────────
        cur.execute(
            "SELECT COUNT(DISTINCT user_id) AS cnt FROM query_logs"
        )
        active_users = cur.fetchone()["cnt"]

        queries_per_user = (
            round(total_queries / active_users, 2)
            if active_users > 0 else 0.0
        )

        # ── Document metrics ──────────────────────────────────────────
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM documents WHERE status = 'ready'"
        )
        total_documents = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) AS cnt FROM document_chunks")
        total_chunks = cur.fetchone()["cnt"]

        # ── Tier breakdown ────────────────────────────────────────────
        cur.execute(
            """
            SELECT tier, COUNT(*) AS cnt
            FROM query_logs
            WHERE tier IS NOT NULL
            GROUP BY tier ORDER BY cnt DESC
            """
        )
        tier_breakdown = {r["tier"]: r["cnt"] for r in cur.fetchall()}

        # ── Recent latency trend (last 24h, hourly buckets) ──────────
        cur.execute(
            """
            SELECT date_trunc('hour', created_at) AS hour,
                   COUNT(*) AS queries,
                   AVG(latency_ms)::real AS avg_latency_ms
            FROM query_logs
            WHERE created_at > NOW() - INTERVAL '24 hours'
              AND latency_ms IS NOT NULL
            GROUP BY hour
            ORDER BY hour
            """
        )
        latency_trend = [
            {
                "hour": str(r["hour"]),
                "queries": r["queries"],
                "avg_latency_ms": r["avg_latency_ms"],
            }
            for r in cur.fetchall()
        ]

        return {
            "total_queries": total_queries,
            "average_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(float(p95_latency), 2),
            "hallucination_rate": hallucination_rate,
            "total_feedback": total_feedback,
            "negative_feedback": negative_feedback,
            "retrieval_success_rate": retrieval_success,
            "active_users": active_users,
            "queries_per_user": queries_per_user,
            "total_documents_uploaded": total_documents,
            "total_document_chunks": total_chunks,
            "tier_breakdown": tier_breakdown,
            "latency_trend_24h": latency_trend,
        }
