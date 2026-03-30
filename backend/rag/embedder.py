"""
embedder.py — Generate dense vector embeddings for text chunks.

Uses sentence-transformers ``all-MiniLM-L6-v2`` (384-dim) by default.
The model is loaded once and reused across calls.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
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

    logger.info("Loading embedding model: %s", model_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        _st_logger = logging.getLogger("safetensors")
        _prev = _st_logger.level
        _st_logger.setLevel(logging.ERROR)
        _old_stdout, _old_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            model = SentenceTransformer(model_name)
        finally:
            sys.stdout, sys.stderr = _old_stdout, _old_stderr
            _st_logger.setLevel(_prev)
    logger.info("Embedding model loaded  (dim=%d)", model.get_sentence_embedding_dimension())
    return model


def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encode a list of strings into a 2-D float32 numpy array.

    Parameters
    ----------
    texts : list[str]
        The strings to embed.
    model_name : str
        HuggingFace model identifier.
    batch_size : int
        Encoding batch size.
    show_progress : bool
        Show a tqdm progress bar.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), embedding_dim)``, dtype ``float32``.
    """
    model = _load_model(model_name)
    logger.info("Embedding %d texts with batch_size=%d …", len(texts), batch_size)
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,          # L2-normalize for cosine similarity
    )
    return embeddings.astype(np.float32)


def embed_query(
    query: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Embed a single query string.  Returns shape ``(1, dim)`` float32 array.
    """
    model = _load_model(model_name)
    vec: np.ndarray = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec.astype(np.float32)
