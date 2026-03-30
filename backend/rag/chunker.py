"""
chunker.py — Split legal documents into overlapping token-based chunks.

Each chunk retains the original metadata (act, section, title) plus a
sequential chunk_id so that it can be traced back to its source.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from rag.loader import LegalDocument

logger = logging.getLogger(__name__)


class Chunk(TypedDict):
    """A single text chunk with metadata."""
    chunk_id: str
    act: str
    section: str
    title: str
    text: str


def _split_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    """
    Naively split *text* into segments of roughly *max_tokens* words with
    *overlap* words shared between consecutive segments.

    We use whitespace-split words as a proxy for tokens (good enough for
    English legal text and avoids a tokenizer dependency).
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + max_tokens
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        # Advance by (max_tokens - overlap) to create the overlap window
        start += max_tokens - overlap
    return chunks


def chunk_documents(
    documents: list[LegalDocument],
    max_tokens: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Convert a list of legal documents into overlapping text chunks.

    Parameters
    ----------
    documents : list[LegalDocument]
        Output of :func:`loader.load_all_files`.
    max_tokens : int
        Maximum number of whitespace-split tokens per chunk.
    overlap : int
        Number of overlapping tokens between consecutive chunks.

    Returns
    -------
    list[Chunk]
        Flat list of chunks with metadata.
    """
    all_chunks: list[Chunk] = []
    global_id = 0

    for doc in documents:
        # Build a rich text block that includes metadata for better retrieval
        full_text = (
            f"{doc['act']} — {doc['section']}: {doc['title']}.\n"
            f"{doc['text']}"
        )
        segments = _split_text(full_text, max_tokens, overlap)

        for idx, seg in enumerate(segments):
            all_chunks.append(
                Chunk(
                    chunk_id=f"chunk_{global_id}",
                    act=doc["act"],
                    section=doc["section"],
                    title=doc["title"],
                    text=seg,
                )
            )
            global_id += 1

    logger.info(
        "Chunked %d documents into %d chunks (max_tokens=%d, overlap=%d)",
        len(documents),
        len(all_chunks),
        max_tokens,
        overlap,
    )
    return all_chunks
