"""
bm25_search.py — BM25 lexical (keyword) search over text chunks.

Builds an Okapi BM25 index from chunk texts and supports keyword-based
retrieval.  Used alongside dense vector search for hybrid retrieval.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from rag.chunker import Chunk

logger = logging.getLogger(__name__)

_BM25_FILE = "bm25_corpus.json"

# ── Simple legal-domain stop words to remove noise ─────────────────────────
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "shall", "would",
    "should", "may", "might", "can", "could", "of", "in", "to", "for",
    "with", "on", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "either", "neither", "this", "that", "these", "those",
    "it", "its", "he", "she", "him", "her", "his", "they", "them",
    "their", "we", "us", "our", "i", "me", "my", "you", "your",
    "which", "who", "whom", "what", "where", "when", "how", "if",
    "than", "such", "no", "any", "each", "every", "all", "some",
    "also", "other", "more", "very", "only", "just", "about",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop words."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


class BM25Index:
    """
    Wrapper around ``rank_bm25.BM25Okapi`` that stores the tokenized corpus
    and supports persistence.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._corpus_tokens: list[list[str]] = [_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info("BM25 index built: %d documents", len(chunks))

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """
        Return ``(chunk_index, bm25_score)`` pairs, sorted descending.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        return results

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, index_dir: str | Path) -> None:
        """Save the tokenized corpus (BM25 is rebuilt on load)."""
        out = Path(index_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / _BM25_FILE, "w", encoding="utf-8") as fh:
            json.dump(self._corpus_tokens, fh)
        logger.info("BM25 corpus saved to %s", out / _BM25_FILE)

    @classmethod
    def load(cls, chunks: list[Chunk], index_dir: str | Path) -> "BM25Index":
        """
        Load a previously-saved tokenized corpus.  Falls back to rebuilding
        from *chunks* if the file is missing.
        """
        corpus_file = Path(index_dir) / _BM25_FILE
        instance = cls.__new__(cls)
        instance._chunks = chunks

        if corpus_file.exists():
            with open(corpus_file, "r", encoding="utf-8") as fh:
                instance._corpus_tokens = json.load(fh)
            logger.info("BM25 corpus loaded from %s", corpus_file)
        else:
            logger.warning("BM25 corpus not found — rebuilding from chunks")
            instance._corpus_tokens = [_tokenize(c["text"]) for c in chunks]

        instance._bm25 = BM25Okapi(instance._corpus_tokens)
        logger.info("BM25 index ready: %d documents", len(chunks))
        return instance
