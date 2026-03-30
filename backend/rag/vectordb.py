"""
vectordb.py — Qdrant Cloud vector store: build, upsert, load, and search.

Uses Qdrant Cloud instead of local FAISS.  A local ``metadata.json``
sidecar is still written so that chunk metadata can be loaded quickly
at startup without scrolling the entire collection.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.chunker import Chunk

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
COLLECTION = "astralex_legal_chunks"
_META_FILE = "metadata.json"


@dataclass
class VectorIndex:
    """Thin handle around a Qdrant collection."""
    client: QdrantClient
    collection: str
    ntotal: int = 0


def _get_client() -> QdrantClient:
    """Create a Qdrant client from environment variables."""
    url = os.environ["QDRANT_URL"]
    api_key = os.environ["QDRANT_API_KEY"]
    return QdrantClient(url=url, api_key=api_key)


# ── Build / save ───────────────────────────────────────────────────────────

def build_index(
    embeddings: np.ndarray,
    collection: str = COLLECTION,
) -> VectorIndex:
    """
    (Re-)create the Qdrant collection for the given embedding dimension.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n, dim)``, dtype ``float32``.
    collection : str
        Target Qdrant collection name.

    Returns
    -------
    VectorIndex
        Handle ready for :func:`save_index`.
    """
    client = _get_client()
    dim = int(embeddings.shape[1])
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("Created Qdrant collection '%s' (dim=%d)", collection, dim)
    return VectorIndex(client=client, collection=collection, ntotal=0)


def save_index(
    index: VectorIndex,
    chunks: list[Chunk],
    index_dir: str | Path,
    embeddings: np.ndarray | None = None,
) -> None:
    """
    Upsert vectors + payloads to Qdrant and save local metadata backup.

    Parameters
    ----------
    index : VectorIndex
        Handle returned by :func:`build_index` or :func:`load_index`.
    chunks : list[Chunk]
        Chunk dicts (stored as Qdrant payloads).
    index_dir : str | Path
        Local directory for the metadata sidecar.
    embeddings : np.ndarray | None
        If provided, upsert to Qdrant.  Pass ``None`` to only write
        the local metadata file.
    """
    out = Path(index_dir)
    out.mkdir(parents=True, exist_ok=True)

    if embeddings is not None:
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            points = [
                PointStruct(
                    id=j,
                    vector=embeddings[j].tolist(),
                    payload=chunks[j],
                )
                for j in range(i, end)
            ]
            index.client.upsert(
                collection_name=index.collection, points=points,
            )
        index.ntotal = len(chunks)
        logger.info(
            "Upserted %d vectors to Qdrant collection '%s'",
            len(chunks), index.collection,
        )

    with open(out / _META_FILE, "w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    logger.info("Saved metadata (%d chunks) to %s", len(chunks), out / _META_FILE)


# ── Load ───────────────────────────────────────────────────────────────────

def load_index(index_dir: str | Path) -> tuple[VectorIndex, list[Chunk]]:
    """
    Connect to Qdrant and load local metadata.

    Returns
    -------
    (VectorIndex, chunks)
    """
    meta_file = Path(index_dir) / _META_FILE
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_file}")

    with open(meta_file, "r", encoding="utf-8") as fh:
        chunks: list[Chunk] = json.load(fh)

    client = _get_client()
    info = client.get_collection(COLLECTION)
    ntotal = info.points_count or 0

    logger.info(
        "Connected to Qdrant collection '%s': %d vectors", COLLECTION, ntotal,
    )
    logger.info("Loaded metadata: %d chunks", len(chunks))

    return VectorIndex(client=client, collection=COLLECTION, ntotal=ntotal), chunks


# ── Search ─────────────────────────────────────────────────────────────────

def search_index(
    index: VectorIndex,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the Qdrant collection.

    Parameters
    ----------
    index : VectorIndex
    query_embedding : np.ndarray
        Shape ``(1, dim)``.
    top_k : int

    Returns
    -------
    (scores, indices) — both shape ``(1, top_k)``.
        Matches the interface previously provided by FAISS so that callers
        (pipeline, retrieval_loop) need no changes.
    """
    results = index.client.query_points(
        collection_name=index.collection,
        query=query_embedding[0].tolist(),
        limit=top_k,
    )

    scores: list[float] = []
    indices: list[int] = []
    for point in results.points:
        scores.append(float(point.score))
        indices.append(int(point.id))

    # Pad to top_k (mirrors FAISS behaviour for unfilled slots)
    while len(scores) < top_k:
        scores.append(0.0)
        indices.append(-1)

    return (
        np.array([scores], dtype=np.float32),
        np.array([indices], dtype=np.int64),
    )
