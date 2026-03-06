"""
reranker.py — Cross-encoder reranking of retrieved candidate chunks.

After hybrid retrieval returns ~20-40 candidates, this module reranks them
using a cross-encoder model (``cross-encoder/ms-marco-MiniLM-L-6-v2``) that
jointly encodes (query, chunk) pairs to produce a relevance score.

Cross-encoder reranking dramatically improves precision over bi-encoder
similarity alone.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag.chunker import Chunk

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    """Load and cache the cross-encoder model."""
    import io
    import os
    import sys
    import warnings

    # Suppress progress bars and verbose logging from transformers/HF
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except ImportError:
        pass

    logger.info("Loading cross-encoder: %s", model_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        _st_logger = logging.getLogger("safetensors")
        _prev = _st_logger.level
        _st_logger.setLevel(logging.ERROR)
        _old_stdout, _old_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            model = CrossEncoder(model_name, max_length=512)
        finally:
            sys.stdout, sys.stderr = _old_stdout, _old_stderr
            _st_logger.setLevel(_prev)
    logger.info("Cross-encoder loaded")
    return model


def rerank(
    query: str,
    chunks: list[Chunk],
    scores: list[float] | None = None,
    model_name: str = _DEFAULT_MODEL,
    top_k: int = 5,
) -> list[tuple[Chunk, float]]:
    """
    Rerank *chunks* by relevance to *query* using a cross-encoder.

    Parameters
    ----------
    query : str
        The search query (preferably the rewritten query).
    chunks : list[Chunk]
        Candidate chunks from hybrid retrieval.
    scores : list[float] | None
        Original retrieval scores (unused — only for logging/debug).
    model_name : str
        HuggingFace cross-encoder model identifier.
    top_k : int
        How many top results to return after reranking.

    Returns
    -------
    list[tuple[Chunk, float]]
        Top-k ``(chunk, cross_encoder_score)`` pairs, sorted descending.
    """
    if not chunks:
        return []

    model = _load_cross_encoder(model_name)

    # Build (query, document) pairs for cross-encoder
    pairs = [(query, chunk["text"]) for chunk in chunks]

    # Score all pairs
    ce_scores: list[float] = model.predict(pairs).tolist()

    # Zip, sort, and return top_k
    scored = list(zip(chunks, ce_scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    top = scored[:top_k]
    logger.info(
        "Reranked %d candidates → top %d (best=%.4f, worst=%.4f)",
        len(chunks),
        len(top),
        top[0][1] if top else 0,
        top[-1][1] if top else 0,
    )
    return top
