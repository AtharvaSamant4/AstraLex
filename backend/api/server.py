"""
server.py — FastAPI REST API for the Legal RAG Chatbot.

ChatGPT-style stateful conversational backend with:
  • User authentication (signup / login  — bcrypt + JWT)
  • Persistent per-user chat sessions (Neon PostgreSQL)
  • Conversation memory (last N messages injected into RAG pipeline)
  • Context-aware retrieval & query rewriting
  • Streaming (SSE) and synchronous endpoints
  • Document upload / management (PDF, DOCX, TXT)
  • Query analytics & system monitoring metrics

Auth endpoints (public)
-----------------------
POST /auth/signup       create account
POST /auth/login        obtain JWT

Chat endpoints (JWT-protected)
------------------------------
POST   /chat/sessions              create a new session
GET    /chat/sessions              list user's sessions
GET    /chat/sessions/{id}         session detail + messages
DELETE /chat/sessions/{id}         delete session
PATCH  /chat/sessions/{id}         rename session
POST   /chat/sessions/{id}/message synchronous chat (full pipeline)
POST   /chat/sessions/{id}/stream  streaming chat (SSE)
POST   /chat/sessions/{id}/feedback  thumbs-up/down

Document endpoints (JWT-protected)
----------------------------------
POST   /documents/upload           upload a document (PDF/DOCX/TXT)
GET    /documents                  list user's documents
DELETE /documents/{id}             delete a document

Analytics / Monitoring (JWT-protected)
--------------------------------------
GET  /metrics                       system monitoring metrics
GET  /analytics/top-queries         most common queries
GET  /analytics/failure-queries     failed / low-quality queries
GET  /analytics/retrieval-performance  retrieval success & distribution

Utility (public)
----------------
GET  /health
GET  /stats
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated, AsyncGenerator

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, Field

from auth.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from database.crud import (
    create_user,
    get_user_by_email,
    save_feedback,
    get_stats,
    create_document,
    list_documents,
    list_session_documents,
    get_document,
    delete_document,
    create_password_reset_token,
    get_valid_reset_token,
    mark_reset_token_used,
    update_user_password,
)
from database.schema import init_schema
from documents.processor import (
    process_document_async,
    invalidate_user_index,
)
from analytics.query_analytics import (
    top_queries as analytics_top_queries,
    failure_queries as analytics_failure_queries,
    retrieval_performance as analytics_retrieval_performance,
)
from metrics.metrics_service import get_system_metrics
from services.chat_service import (
    chat as svc_chat,
    chat_stream as svc_chat_stream,
    create_session as svc_create_session,
    list_sessions as svc_list_sessions,
    get_session_detail as svc_get_session_detail,
    delete_session as svc_delete_session,
    _load_chatbot_if_needed,
)
from database.crud import update_session_title

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)


# ── FastAPI lifespan ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    logger.info("🚀  Starting Legal RAG Chatbot API …")
    _load_chatbot_if_needed()
    try:
        init_schema()
        logger.info("✅  Database schema ready (Neon PostgreSQL)")
    except Exception as exc:
        logger.warning("⚠️  Database unavailable — running without persistence: %s", exc)
    logger.info("✅  Chatbot loaded and ready")
    yield
    logger.info("🛑  Shutting down")


app = FastAPI(
    title="Indian Legal RAG Chatbot",
    description=(
        "ChatGPT-style conversational AI backend — "
        "user auth · persistent sessions · conversation memory · "
        "document upload RAG · query analytics · system monitoring · "
        "deep-research agentic RAG pipeline for Indian law."
    ),
    version="5.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shorthand for the auth dependency
CurrentUser = Annotated[dict, Depends(get_current_user)]

# Allowed file types for upload
_ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}


# ═══════════════════════════════════════════════════════════════════════════
# Schemas
# ═══════════════════════════════════════════════════════════════════════════

class SignupRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=255, examples=["user@example.com"])
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(..., examples=["user@example.com"])
    password: str = Field(...)


class AuthResponse(BaseModel):
    token: str
    user_id: int
    email: str


class SessionCreate(BaseModel):
    title: str | None = Field(None, max_length=200, examples=["IPC discussion"])


class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., examples=["user@example.com"])


class ResetPasswordRequest(BaseModel):
    token: str = Field(...)
    new_password: str = Field(..., min_length=6, max_length=128)


class SessionRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class ChatMessageRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=2000,
        examples=["What is the punishment for murder under IPC?"],
    )


class ChatMessageResponse(BaseModel):
    answer: str
    sources: list[str]
    rewritten_query: str
    verified: bool
    timings: dict[str, float] = {}
    research_plan: dict | None = None
    evidence_graph_stats: dict[str, int] | None = None
    complexity: str = "simple"
    retrieval_iterations: int = 1
    follow_up_queries: list[str] = []
    tier: str = "standard"
    user_message_id: int = 0
    assistant_message_id: int = 0


class FeedbackRequest(BaseModel):
    message_id: int
    rating: int = Field(..., ge=-1, le=1, description="-1 = bad, 1 = good")
    comment: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Auth routes (public)
# ═══════════════════════════════════════════════════════════════════════════

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@app.post("/auth/signup", response_model=AuthResponse, status_code=201)
async def signup(req: SignupRequest):
    """Create a new user account and return a JWT."""
    if not _EMAIL_RE.match(req.email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    existing = get_user_by_email(req.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    try:
        hashed = hash_password(req.password)
        user = create_user(req.email, hashed)
        token = create_access_token(user["id"], user["email"])
        return AuthResponse(token=token, user_id=user["id"], email=user["email"])
    except Exception as exc:
        logger.exception("Signup error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/auth/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """Authenticate and return a JWT."""
    user = get_user_by_email(req.email)
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token(user["id"], user["email"])
    return AuthResponse(token=token, user_id=user["id"], email=user["email"])


@app.post("/auth/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    """Generate a password reset token (always returns 200 to prevent email enumeration)."""
    user = get_user_by_email(req.email)
    if user:
        token = secrets.token_urlsafe(32)
        expires = datetime.now(timezone.utc) + timedelta(hours=1)
        try:
            create_password_reset_token(user["id"], token, expires)
            logger.info("Password reset token created for user %s (token: %s)", user["id"], token)
        except Exception:
            logger.exception("Failed to create password reset token")
    return {"message": "If an account exists with that email, a reset link has been generated."}


@app.post("/auth/reset-password")
async def reset_password(req: ResetPasswordRequest):
    """Reset password using a valid token."""
    token_record = get_valid_reset_token(req.token)
    if not token_record:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    try:
        hashed = hash_password(req.new_password)
        update_user_password(token_record["user_id"], hashed)
        mark_reset_token_used(req.token)
        return {"message": "Password reset successful"}
    except Exception as exc:
        logger.exception("Password reset error")
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# Session routes (JWT-protected)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/chat/sessions", status_code=201)
async def create_new_session(req: SessionCreate | None = None, user: CurrentUser = None):
    """Create a new chat session for the authenticated user."""
    try:
        title = req.title if req else None
        session = svc_create_session(user["id"], title)
        return session
    except Exception as exc:
        logger.exception("Session create error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/chat/sessions")
async def list_user_sessions(limit: int = 20, user: CurrentUser = None):
    """List the authenticated user's recent sessions."""
    try:
        sessions = svc_list_sessions(user["id"], limit)
        return {"sessions": sessions}
    except Exception as exc:
        logger.exception("Session list error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/chat/sessions/{session_id}")
async def get_session_detail_route(session_id: str, user: CurrentUser = None):
    """Get a session with all its messages (ownership-checked)."""
    try:
        detail = svc_get_session_detail(session_id, user["id"])
        if not detail:
            raise HTTPException(status_code=404, detail="Session not found")
        return detail
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Session detail error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/chat/sessions/{session_id}")
async def delete_session_route(session_id: str, user: CurrentUser = None):
    """Delete a session and all its messages."""
    try:
        deleted = svc_delete_session(session_id, user["id"])
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "deleted", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Session delete error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.patch("/chat/sessions/{session_id}")
async def rename_session_route(
    session_id: str, req: SessionRename, user: CurrentUser = None
):
    """Rename a session."""
    try:
        ok = update_session_title(session_id, user["id"], req.title)
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "renamed", "session_id": session_id, "title": req.title}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Session rename error")
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# Chat message routes (JWT-protected)
# ═══════════════════════════════════════════════════════════════════════════

@app.post(
    "/chat/sessions/{session_id}/message",
    response_model=ChatMessageResponse,
)
async def chat_message(
    session_id: str, req: ChatMessageRequest, user: CurrentUser = None,
):
    """
    Synchronous chat — runs the full deep-research agentic pipeline with
    conversation memory loaded from the database.
    """
    try:
        result = svc_chat(session_id, user["id"], req.question)
        return ChatMessageResponse(
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
            tier=result.tier,
            user_message_id=result.user_message_id,
            assistant_message_id=result.assistant_message_id,
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as exc:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/sessions/{session_id}/stream")
async def chat_stream_route(
    session_id: str, req: ChatMessageRequest, user: CurrentUser = None,
):
    """
    Streaming chat (SSE) — pipeline + conversation memory, tokens streamed.
    """
    return StreamingResponse(
        svc_chat_stream(session_id, user["id"], req.question),
        media_type="text/event-stream",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Feedback (JWT-protected)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/chat/sessions/{session_id}/feedback")
async def submit_feedback(
    session_id: str, req: FeedbackRequest, user: CurrentUser = None,
):
    """Submit thumbs-up/down feedback for a message."""
    try:
        fb_id = save_feedback(req.message_id, req.rating, req.comment)
        return {"feedback_id": fb_id, "status": "saved"}
    except Exception as exc:
        logger.exception("Feedback error")
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# Document upload & management (JWT-protected)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/documents/upload", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Query(None, max_length=200),
    session_id: str | None = Query(None, max_length=100),
    user: CurrentUser = None,
):
    """
    Upload a document (PDF / DOCX / TXT).
    Optionally attach to a session via session_id (ownership-verified).
    Processing happens asynchronously — returns immediately with status 202.
    """
    # If session_id provided, verify the session belongs to the user
    if session_id:
        from database.crud import get_session
        sess = get_session(session_id, user["id"])
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")

    # Validate extension
    filename = file.filename or "untitled"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    # Read bytes (limit: 20 MB)
    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Create DB record
    try:
        doc = create_document(
            user_id=user["id"],
            filename=filename,
            file_type=ext,
            title=title,
            session_id=session_id,
        )
    except Exception as exc:
        logger.exception("Document create error")
        raise HTTPException(status_code=500, detail=str(exc))

    # Fire background processing
    process_document_async(doc["id"], user["id"], file_bytes, ext)

    return {
        "document_id": doc["id"],
        "filename": doc["filename"],
        "status": doc["status"],
        "session_id": session_id,
        "message": "Document uploaded — processing in background.",
    }


@app.get("/documents")
async def list_user_documents(
    session_id: str | None = Query(None),
    user: CurrentUser = None,
):
    """List documents — optionally filtered by session_id."""
    try:
        if session_id:
            docs = list_session_documents(session_id, user["id"])
        else:
            docs = list_documents(user["id"])
        return {"documents": docs}
    except Exception as exc:
        logger.exception("Document list error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/documents/{document_id}")
async def delete_user_document(document_id: int, user: CurrentUser = None):
    """Delete a document and invalidate the user's document index."""
    doc = get_document(document_id, user["id"])
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        delete_document(document_id, user["id"])
        invalidate_user_index(user["id"])
        return {"status": "deleted", "document_id": document_id}
    except Exception as exc:
        logger.exception("Document delete error")
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# System Monitoring Metrics (JWT-protected)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/metrics")
async def metrics_endpoint(user: CurrentUser = None):
    """Return comprehensive system monitoring metrics."""
    try:
        data = get_system_metrics()
        return data
    except Exception as exc:
        logger.exception("Metrics error")
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# Admin Analytics (JWT-protected)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/analytics/top-queries")
async def analytics_top_queries_route(
    limit: int = Query(20, ge=1, le=100),
    user: CurrentUser = None,
):
    """Return the most frequently asked queries."""
    try:
        return {"queries": analytics_top_queries(limit)}
    except Exception as exc:
        logger.exception("Analytics top-queries error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/analytics/failure-queries")
async def analytics_failure_queries_route(
    limit: int = Query(20, ge=1, le=100),
    user: CurrentUser = None,
):
    """Return queries that failed — no sources or negative feedback."""
    try:
        return {"queries": analytics_failure_queries(limit)}
    except Exception as exc:
        logger.exception("Analytics failure-queries error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/analytics/retrieval-performance")
async def analytics_retrieval_performance_route(user: CurrentUser = None):
    """Return aggregate retrieval performance metrics."""
    try:
        return analytics_retrieval_performance()
    except Exception as exc:
        logger.exception("Analytics retrieval-performance error")
        raise HTTPException(status_code=500, detail=str(exc))


# ═══════════════════════════════════════════════════════════════════════════
# Utility (public)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "5.0.0",
        "pipeline": "deep-research-agentic-rag",
        "auth": "jwt",
        "features": [
            "document-upload",
            "query-analytics",
            "system-metrics",
        ],
    }


@app.get("/stats")
async def usage_stats():
    try:
        stats = get_stats()
        return stats
    except Exception as exc:
        logger.exception("Stats error")
        raise HTTPException(status_code=500, detail=str(exc))
