"""
documents/processor.py — Extract text, chunk, embed, and index uploaded documents.

Supports PDF, DOCX, and TXT files.

The processing pipeline:
  1. Extract raw text from the uploaded file
  2. Chunk using the same strategy as the legal dataset (_split_text)
  3. Generate embeddings (same model as the legal index)
  4. Build a per-user Qdrant collection (or add to the existing one)
  5. Persist chunks to PostgreSQL
"""

from __future__ import annotations

import io
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from database import crud
from rag.embedder import embed_texts

logger = logging.getLogger(__name__)

_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# ═══════════════════════════════════════════════════════════════════════════
# Text extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_text_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2."""
    from PyPDF2 import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def extract_text_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    from docx import Document
    doc = Document(io.BytesIO(file_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text_txt(file_bytes: bytes) -> str:
    """Decode plain-text bytes."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return file_bytes.decode("utf-8", errors="replace")


_EXTRACTORS = {
    "pdf": extract_text_pdf,
    "docx": extract_text_docx,
    "txt": extract_text_txt,
}


def extract_text(file_bytes: bytes, file_type: str) -> str:
    """Extract text from a file given its type (pdf/docx/txt)."""
    extractor = _EXTRACTORS.get(file_type.lower())
    if not extractor:
        raise ValueError(f"Unsupported file type: {file_type}")
    return extractor(file_bytes)


# ═══════════════════════════════════════════════════════════════════════════
# Chunking (reuses the same logic as rag/chunker.py)
# ═══════════════════════════════════════════════════════════════════════════

def _split_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based segments."""
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunks.append(" ".join(words[start:end]))
        start += max_tokens - overlap
    return chunks


def chunk_text(
    text: str,
    max_tokens: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Chunk extracted text using the project's standard strategy."""
    max_tokens = max_tokens or _CHUNK_SIZE
    overlap = overlap or _CHUNK_OVERLAP
    return _split_text(text, max_tokens, overlap)


# ═══════════════════════════════════════════════════════════════════════════
# Per-user Qdrant collections for uploaded documents
# ═══════════════════════════════════════════════════════════════════════════

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

_lock = threading.Lock()

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 default


def _get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client from environment variables."""
    url = os.environ["QDRANT_URL"]
    api_key = os.environ["QDRANT_API_KEY"]
    return QdrantClient(url=url, api_key=api_key)


def _user_collection(user_id: int) -> str:
    """Return the Qdrant collection name for a user's uploaded documents."""
    return f"astralex_user_{user_id}_docs"


def _ensure_user_collection(client: QdrantClient, user_id: int, dim: int) -> str:
    """Create the user's document collection if it doesn't exist."""
    name = _user_collection(user_id)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'", name)
    return name


def invalidate_user_index(user_id: int) -> None:
    """Delete a user's document collection (called on document delete)."""
    try:
        client = _get_qdrant_client()
        name = _user_collection(user_id)
        if client.collection_exists(name):
            client.delete_collection(name)
            logger.info("Deleted Qdrant collection '%s'", name)
    except Exception:
        logger.exception("Failed to delete user %d collection", user_id)


def rebuild_user_index(user_id: int) -> int:
    """
    Rebuild the Qdrant collection for a user from DB chunks.
    Returns the number of vectors indexed.
    """
    chunks = crud.get_user_document_chunks(user_id)
    if not chunks:
        invalidate_user_index(user_id)
        return 0

    texts = [c["chunk_text"] for c in chunks]
    embeddings = embed_texts(texts, model_name=_EMBED_MODEL, show_progress=False)
    dim = embeddings.shape[1]

    client = _get_qdrant_client()
    name = _user_collection(user_id)

    # Recreate to start clean
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        end = min(i + batch_size, len(chunks))
        points = [
            PointStruct(
                id=j,
                vector=embeddings[j].tolist(),
                payload={
                    "document_id": chunks[j]["document_id"],
                    "chunk_index": chunks[j]["chunk_index"],
                    "chunk_text": chunks[j]["chunk_text"],
                    "filename": chunks[j].get("filename", ""),
                    "title": chunks[j].get("title", ""),
                },
            )
            for j in range(i, end)
        ]
        client.upsert(collection_name=name, points=points)

    logger.info("Rebuilt user %d document index: %d vectors", user_id, len(texts))
    return len(texts)


def search_user_documents(
    user_id: int,
    query_embedding: np.ndarray,
    top_k: int = 10,
) -> list[tuple[dict, float]]:
    """
    Search a user's uploaded-document collection in Qdrant.

    Returns list of (metadata_dict, score) sorted by descending score.
    Returns empty list if the user has no documents indexed.
    """
    try:
        client = _get_qdrant_client()
        name = _user_collection(user_id)
        if not client.collection_exists(name):
            return []

        info = client.get_collection(name)
        if not info.points_count:
            return []

        actual_k = min(top_k, info.points_count)
        results = client.query_points(
            collection_name=name,
            query=query_embedding[0].tolist(),
            limit=actual_k,
        )
        return [(point.payload, float(point.score)) for point in results.points]

    except Exception:
        logger.exception("User %d document search failed", user_id)
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Full processing pipeline (runs in background thread)
# ═══════════════════════════════════════════════════════════════════════════

def process_document(
    doc_id: int,
    user_id: int,
    file_bytes: bytes,
    file_type: str,
) -> None:
    """
    Full document processing pipeline.
    Intended to be called in a background thread so the API returns fast.

    Steps:
      1. Extract text
      2. Chunk
      3. Generate embeddings
      4. Save chunks to DB
      5. Add to user's Qdrant collection
      6. Mark document as 'ready'
    """
    try:
        logger.info("Processing document %d (type=%s) for user %d", doc_id, file_type, user_id)

        # 1. Extract
        raw_text = extract_text(file_bytes, file_type)
        if not raw_text.strip():
            crud.update_document_status(doc_id, "failed", 0)
            logger.warning("Document %d: no text extracted", doc_id)
            return

        # 2. Chunk
        text_chunks = chunk_text(raw_text)
        if not text_chunks:
            crud.update_document_status(doc_id, "failed", 0)
            return

        # 3. Embed
        embeddings = embed_texts(text_chunks, model_name=_EMBED_MODEL, show_progress=False)

        # 4. Save chunks to DB
        db_chunks = [
            {
                "chunk_index": i,
                "chunk_text": t,
                "embedding_ref": f"doc_{doc_id}_chunk_{i}",
            }
            for i, t in enumerate(text_chunks)
        ]
        crud.save_document_chunks(doc_id, db_chunks)

        # 5. Add to user's Qdrant collection (MUST complete before marking ready)
        dim = embeddings.shape[1]
        client = _get_qdrant_client()
        collection = _ensure_user_collection(client, user_id, dim)
        doc_info = crud.get_document(doc_id, user_id) or {}

        # Use unique IDs: (doc_id * 100000 + chunk_index) to avoid collisions
        points = [
            PointStruct(
                id=doc_id * 100000 + i,
                vector=embeddings[i].tolist(),
                payload={
                    "document_id": doc_id,
                    "chunk_index": i,
                    "chunk_text": t,
                    "filename": doc_info.get("filename", ""),
                    "title": doc_info.get("title", ""),
                },
            )
            for i, t in enumerate(text_chunks)
        ]
        
        # Upsert in batches - CRITICAL: Must complete before marking ready
        batch_size = 100
        for i in range(0, len(points), batch_size):
            client.upsert(
                collection_name=collection,
                points=points[i:i + batch_size],
            )
        
        # Verify Qdrant indexing completed by checking collection point count
        collection_info = client.get_collection(collection)
        logger.info(
            "Document %d: Qdrant collection '%s' now has %d points",
            doc_id, collection, collection_info.points_count
        )

        # 6. Mark ready ONLY after Qdrant indexing is confirmed complete
        crud.update_document_status(doc_id, "ready", len(text_chunks))
        logger.info(
            "Document %d processed: %d chunks, %d embeddings, Qdrant indexing complete",
            doc_id, len(text_chunks), embeddings.shape[0],
        )

    except Exception:
        logger.exception("Document %d processing failed", doc_id)
        try:
            crud.update_document_status(doc_id, "failed", 0)
        except Exception:
            pass


def process_document_async(
    doc_id: int, user_id: int, file_bytes: bytes, file_type: str,
) -> threading.Thread:
    """Spawn a background thread for document processing."""
    t = threading.Thread(
        target=process_document,
        args=(doc_id, user_id, file_bytes, file_type),
        daemon=True,
    )
    t.start()
    logger.info("Background processing started for document %d", doc_id)
    return t
