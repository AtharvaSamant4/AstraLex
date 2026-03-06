"""
vectordb.py — FAISS vector store: build, save, load, and search.

Stores the FAISS index alongside a JSON metadata sidecar so that each
vector can be mapped back to its chunk text and legal source.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag.chunker import Chunk

logger = logging.getLogger(__name__)

# ── File names inside the index directory ──────────────────────────────────
_INDEX_FILE = "faiss.index"
_META_FILE = "metadata.json"


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS flat inner-product index from the given embeddings.

    Because embeddings are L2-normalised in the embedder, inner-product
    search is equivalent to cosine similarity.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n, dim)``, dtype ``float32``.

    Returns
    -------
    faiss.IndexFlatIP
        Populated FAISS index.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("Built FAISS index: %d vectors, dim=%d", index.ntotal, dim)
    return index


def save_index(
    index: faiss.IndexFlatIP,
    chunks: list[Chunk],
    index_dir: str | Path,
) -> None:
    """Persist the FAISS index and chunk metadata to disk."""
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / _INDEX_FILE))
    logger.info("Saved FAISS index to %s", out / _INDEX_FILE)

    with open(out / _META_FILE, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    logger.info("Saved metadata (%d chunks) to %s", len(chunks), out / _META_FILE)


def load_index(index_dir: str | Path) -> tuple[faiss.IndexFlatIP, list[Chunk]]:
    """
    Load a previously saved FAISS index and its metadata.

    Returns
    -------
    (index, chunks)
    """
    idx_path = Path(index_dir)

    index_file = idx_path / _INDEX_FILE
    meta_file = idx_path / _META_FILE

    if not index_file.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_file}")

    index = faiss.read_index(str(index_file))
    logger.info("Loaded FAISS index: %d vectors", index.ntotal)

    with open(meta_file, "r", encoding="utf-8") as fh:
        chunks: list[Chunk] = json.load(fh)
    logger.info("Loaded metadata: %d chunks", len(chunks))

    return index, chunks


def search_index(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the FAISS index.

    Parameters
    ----------
    index : faiss.IndexFlatIP
    query_embedding : np.ndarray
        Shape ``(1, dim)``.
    top_k : int

    Returns
    -------
    (scores, indices) — both shape ``(1, top_k)``.
    """
    scores, indices = index.search(query_embedding, top_k)
    return scores, indices
