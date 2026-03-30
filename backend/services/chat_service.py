"""
services/chat_service.py — Stateful conversation orchestrator.

Bridges the RAG pipeline with the database layer so that:
  1. Previous messages are loaded from PostgreSQL
  2. The pipeline's in-memory ``_history`` is hydrated before every call
  3. Both user & assistant messages are persisted after generation
  4. Sessions are auto-titled from the first user question
  5. Every query is logged for analytics
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

from database import crud
from chatbot.chatbot import LegalChatbot, ChatResponse
from analytics.query_analytics import log_chat_query
from documents.processor import search_user_documents, rebuild_user_index
from rag.embedder import embed_query

logger = logging.getLogger(__name__)

# How many prior messages to inject into the RAG pipeline's memory window
_MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "10"))

# ── Lazy singleton for the heavy chatbot ────────────────────────────────────
_chatbot: LegalChatbot | None = None


def _get_chatbot() -> LegalChatbot:
    global _chatbot
    if _chatbot is None:
        from chatbot.chatbot import LegalChatbot
        index_dir = os.getenv("INDEX_DIR", "index")
        _chatbot = LegalChatbot(index_dir=index_dir)
    return _chatbot


def _load_chatbot_if_needed() -> None:
    """Pre-load the chatbot (called at startup)."""
    _get_chatbot()


# ═══════════════════════════════════════════════════════════════════════════
# History hydration
# ═══════════════════════════════════════════════════════════════════════════

def _hydrate_pipeline_history(session_id: str) -> None:
    """
    Load the last *_MEMORY_WINDOW* messages from PostgreSQL and inject
    them into the pipeline's ``_history`` list so that query rewriting
    and follow-up detection work correctly.
    """
    bot = _get_chatbot()
    pipeline = bot._pipeline  # direct access to RAGPipeline

    # Clear stale in-memory history
    pipeline.clear_history()

    recent = crud.get_recent_messages(session_id, n=_MEMORY_WINDOW)
    for msg in recent:
        pipeline._history.append({
            "role": msg["role"],
            "text": msg["content"],
        })

    logger.debug(
        "Hydrated pipeline with %d messages from session %s",
        len(recent), session_id,
    )


# ── User-document context injection ────────────────────────────────────

_DOC_CONTEXT_TOP_K = int(os.getenv("DOC_CONTEXT_TOP_K", "5"))
_DOC_SCORE_THRESHOLD = float(os.getenv("DOC_SCORE_THRESHOLD", "0.25"))


def _get_user_doc_context(user_id: int, question: str) -> str:
    """
    Search the user's uploaded documents for chunks relevant to *question*.
    Returns a formatted context string to inject into the prompt, or empty
    string if nothing relevant is found.
    
    Only searches documents with status='ready' to avoid querying before
    indexing completes.
    """
    try:
        # Check if user has any READY documents before attempting retrieval
        ready_docs = crud.get_ready_documents(user_id)
        if not ready_docs:
            return ""
        
        # Search the user's Qdrant document collection
        qvec = embed_query(question)                    # shape (1, dim)
        results = search_user_documents(user_id, qvec, top_k=_DOC_CONTEXT_TOP_K)

        if not results:
            return ""

        # Filter by score threshold
        relevant = [(meta, sc) for meta, sc in results if sc >= _DOC_SCORE_THRESHOLD]
        if not relevant:
            return ""

        lines = ["[Relevant excerpts from your uploaded documents]"]
        for i, (meta, score) in enumerate(relevant, 1):
            source_label = meta.get("title") or meta.get("filename", "Unknown")
            lines.append(f"\n--- Document: {source_label} (chunk {meta.get('chunk_index', '?')}) ---")
            lines.append(meta.get("chunk_text", "").strip())
        lines.append("\n[End of uploaded document excerpts]\n")
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("User-doc retrieval failed (non-fatal): %s", exc)
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# Session helpers
# ═══════════════════════════════════════════════════════════════════════════

def create_session(user_id: int, title: Optional[str] = None) -> dict:
    """Create a new chat session for a user."""
    return crud.create_session(user_id, title)


def list_sessions(user_id: int, limit: int = 20) -> list[dict]:
    """List recent sessions for a user."""
    return crud.list_sessions(user_id, limit)


def get_session_detail(session_id: str, user_id: int) -> dict | None:
    """Return session metadata + messages + session documents."""
    session = crud.get_session(session_id, user_id)
    if not session:
        return None
    messages = crud.get_messages(session_id)
    documents = crud.list_session_documents(session_id, user_id)
    return {"session": session, "messages": messages, "documents": documents}


def delete_session(session_id: str, user_id: int) -> bool:
    return crud.delete_session(session_id, user_id)


# ═══════════════════════════════════════════════════════════════════════════
# Main chat flow — synchronous
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChatResult:
    """Enriched result returned to the API layer."""
    answer: str
    sources: list[str]
    rewritten_query: str
    verified: bool
    timings: dict[str, float] = field(default_factory=dict)
    research_plan: dict | None = None
    evidence_graph_stats: dict[str, int] | None = None
    complexity: str = "simple"
    retrieval_iterations: int = 1
    follow_up_queries: list[str] = field(default_factory=list)
    tier: str = "standard"
    user_message_id: int = 0
    assistant_message_id: int = 0


def chat(
    session_id: str,
    user_id: int,
    question: str,
) -> ChatResult:
    """
    Full stateful chat flow:
      1. Validate session ownership
      2. Hydrate pipeline history from DB
      3. Run the RAG pipeline
      4. Persist both messages
      5. Auto-title the session if it's the first message
    """
    # 1. Ownership check
    session = crud.get_session(session_id, user_id)
    if not session:
        raise ValueError("Session not found or access denied")

    # 2. Hydrate history
    _hydrate_pipeline_history(session_id)

    # 3. Save user message first (so it appears in DB even if generation fails)
    user_msg_id = crud.save_message(session_id, "user", question)

    # 3b. Retrieve relevant user-document context (if any uploaded)
    doc_context = _get_user_doc_context(user_id, question)
    augmented_question = (
        f"{doc_context}\n\n{question}" if doc_context else question
    )

    # 4. Run pipeline
    bot = _get_chatbot()
    t_start = time.perf_counter()
    result = bot.ask(augmented_question)
    latency_ms = (time.perf_counter() - t_start) * 1000

    # 5. Save assistant response
    assistant_msg_id = crud.save_message(
        session_id, "assistant", result.answer,
        sources=result.sources,
        rewritten_query=result.rewritten_query,
        complexity=result.complexity,
        tier=getattr(result, "tier", None),
        timings=result.timings,
    )

    # 6. Auto-title: use first user question truncated to 80 chars
    if not session.get("title"):
        title = question[:80].strip()
        crud.update_session_title(session_id, user_id, title)

    # 7. Log query for analytics
    try:
        log_chat_query(
            user_id=user_id,
            query=question,
            session_id=session_id,
            rewritten_query=result.rewritten_query,
            retrieved_sources=result.sources,
            latency_ms=latency_ms,
            tier=getattr(result, "tier", None),
            complexity=result.complexity,
            retrieval_chunks_count=len(result.sources),
        )
    except Exception as log_exc:
        logger.warning("Query logging failed (non-fatal): %s", log_exc)

    return ChatResult(
        answer=result.answer,
        sources=result.sources,
        rewritten_query=result.rewritten_query,
        verified=result.verified,
        timings=result.timings,
        research_plan=result.research_plan,
        evidence_graph_stats=result.evidence_graph_stats,
        complexity=result.complexity,
        retrieval_iterations=result.retrieval_iterations,
        follow_up_queries=result.follow_up_queries,
        tier=getattr(result, "tier", "standard") or "standard",
        user_message_id=user_msg_id,
        assistant_message_id=assistant_msg_id,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Streaming chat flow
# ═══════════════════════════════════════════════════════════════════════════

def chat_stream(
    session_id: str,
    user_id: int,
    question: str,
) -> Generator[str, None, None]:
    """
    Streaming variant — yields SSE-formatted tokens.

    After iteration completes, the final metadata events are yielded and
    messages are persisted to the database.
    """
    import json as _json

    # 1. Ownership check
    session = crud.get_session(session_id, user_id)
    if not session:
        yield "event: error\ndata: Session not found\n\n"
        return

    # 2. Hydrate history
    _hydrate_pipeline_history(session_id)

    # 3. Save user message FIRST and get ID
    user_msg_id = crud.save_message(session_id, "user", question)
    
    # 4. Create assistant placeholder message IMMEDIATELY and get ID
    assistant_msg_id = crud.save_message(
        session_id, "assistant", "",
        sources=[],
        rewritten_query=None,
        complexity=None,
        tier=None,
    )
    
    # 5. Send message IDs to frontend IMMEDIATELY (before any tokens)
    yield f"event: message_ids\ndata: {_json.dumps({'user': user_msg_id, 'assistant': assistant_msg_id})}\n\n"

    # 6. Retrieve relevant user-document context (only from READY documents)
    doc_context = _get_user_doc_context(user_id, question)
    augmented_question = (
        f"{doc_context}\n\n{question}" if doc_context else question
    )

    # 7. Stream tokens
    bot = _get_chatbot()
    collected_tokens: list[str] = []
    t_start = time.perf_counter()

    for token in bot.ask_stream(augmented_question):
        # The pipeline yields dict signals for lifecycle events
        # (e.g. {"event": "retry"} when it retries after partial output).
        if isinstance(token, dict):
            event = token.get("event")
            if event == "retry":
                collected_tokens.clear()
                yield "event: retry\ndata: reset\n\n"
            continue
        collected_tokens.append(token)
        yield f"data: {token}\n\n"

    latency_ms = (time.perf_counter() - t_start) * 1000

    # 8. Gather metadata
    sources = bot.get_last_sources()
    rewritten = bot.get_last_rewritten_query()
    complexity = bot.get_last_complexity()
    iterations = bot.get_last_retrieval_iterations()
    graph_stats = bot.get_last_graph_stats()
    follow_ups = bot.get_last_follow_up_queries()
    tier = bot.get_last_tier()

    # 9. Source integrity — suppress sources when the answer indicates
    #     no relevant information was found.  The pipeline sets sources
    #     from retrieved chunks BEFORE generation, so the LLM may hedge
    #     even though chunks were retrieved.
    full_answer = "".join(collected_tokens)
    _NO_INFO_PATTERNS = (
        "no relevant legal documents",
        "not available in our database",
        "couldn't find",
        "could not find",
        "i don't have information",
        "no information was found",
        "i couldn't generate",
        "couldn't generate an answer",
    )
    if any(p in full_answer.lower() for p in _NO_INFO_PATTERNS):
        sources = []

    # 10. UPDATE the existing assistant message (don't create a new one)
    crud.update_message_content(
        assistant_msg_id,
        full_answer,
        sources=sources,
        rewritten_query=rewritten,
        complexity=complexity,
        tier=tier,
    )

    # 11. Auto-title
    if not session.get("title"):
        title = question[:80].strip()
        crud.update_session_title(session_id, user_id, title)

    # 12. Log query for analytics
    try:
        log_chat_query(
            user_id=user_id,
            query=question,
            session_id=session_id,
            rewritten_query=rewritten,
            retrieved_sources=sources,
            latency_ms=latency_ms,
            tier=tier,
            complexity=complexity,
            retrieval_chunks_count=len(sources),
        )
    except Exception as log_exc:
        logger.warning("Query logging failed (non-fatal): %s", log_exc)

    # 13. Send metadata events (message_ids already sent at start)
    yield f"event: tier\ndata: {tier}\n\n"
    yield f"event: rewritten\ndata: {rewritten}\n\n"
    yield f"event: sources\ndata: {'; '.join(sources)}\n\n"
    yield f"event: complexity\ndata: {complexity}\n\n"
    yield f"event: iterations\ndata: {iterations}\n\n"
    if graph_stats:
        yield f"event: graph_stats\ndata: {_json.dumps(graph_stats)}\n\n"
    if follow_ups:
        yield f"event: follow_ups\ndata: {'; '.join(follow_ups)}\n\n"
    yield "event: done\ndata: [DONE]\n\n"
