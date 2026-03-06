"""
retriever.py — High-level retrieval: embed query → search FAISS → return chunks.

Provides a ``Retriever`` class that loads the index once and exposes a
simple ``.retrieve(query)`` method.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from rag.chunker import Chunk
from rag.embedder import embed_query
from rag.vectordb import load_index, search_index

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float

    @property
    def source_label(self) -> str:
        return f"{self.chunk['act']} — {self.chunk['section']}"


class Retriever:
    """
    Stateful retriever: loads the FAISS index once, then answers queries.

    Parameters
    ----------
    index_dir : str | Path
        Folder containing ``faiss.index`` and ``metadata.json``.
    embedding_model : str
        Name of the sentence-transformers model to encode queries.
    top_k : int
        Number of chunks to return per query.
    """

    def __init__(
        self,
        index_dir: str | Path = "index",
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.top_k = top_k

        logger.info("Initializing Retriever (index_dir=%s, top_k=%d)", self.index_dir, self.top_k)
        self._index, self._chunks = load_index(self.index_dir)
        self._cache: dict[str, list[RetrievalResult]] = {}
        logger.info("Retriever ready — %d chunks indexed", len(self._chunks))

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """
        Retrieve the most relevant chunks for the user's *query*.

        Returns a list of ``RetrievalResult`` objects sorted by descending score.
        """

        if query in self._cache:
            logger.debug("Cache hit for query: %s", query[:60])
            return self._cache[query]

        query_vec = embed_query(query, model_name=self.embedding_model)
        scores, indices = search_index(self._index, query_vec, top_k=self.top_k)

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 for unfilled slots
            results.append(RetrievalResult(chunk=self._chunks[idx], score=float(score)))

        # Sort descending by score (should already be, but be explicit)
        results.sort(key=lambda r: r.score, reverse=True)

        # Cache the result
        self._cache[query] = results
        logger.info("Retrieved %d chunks for query: '%s'", len(results), query[:80])
        return results
