#!/usr/bin/env python3
"""
build_index.py — One-time index builder.

Run this ONCE before starting the chatbot or API:

    python build_index.py

It will:
  1. Load all JSON law files from the ``data/`` directory.
  2. Chunk them into overlapping segments.
  3. Generate embeddings with sentence-transformers.
  4. Build and save a FAISS index + metadata to ``index/``.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# ── Bootstrap ──────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger("build_index")


def main() -> None:
    from rag.loader import load_all_files
    from rag.chunker import chunk_documents
    from rag.embedder import embed_texts
    from rag.vectordb import build_index, save_index
    from rag.bm25_search import BM25Index

    data_dir = os.getenv("DATA_DIR", "data")
    index_dir = os.getenv("INDEX_DIR", "index")
    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    t0 = time.perf_counter()

    # ── Step 1: Load ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 / 5 — Loading legal documents from '%s'", data_dir)
    logger.info("=" * 60)
    documents = load_all_files(data_dir)
    logger.info("Loaded %d documents.", len(documents))

    # ── Step 2: Chunk ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 / 5 — Chunking documents (size=%d, overlap=%d)", chunk_size, chunk_overlap)
    logger.info("=" * 60)
    chunks = chunk_documents(documents, max_tokens=chunk_size, overlap=chunk_overlap)
    logger.info("Created %d chunks.", len(chunks))

    # ── Step 3: Embed ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 / 5 — Generating embeddings with '%s'", embedding_model)
    logger.info("=" * 60)
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model_name=embedding_model, show_progress=True)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # ── Step 4: Build & save Qdrant index ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 / 5 — Building Qdrant index and saving to '%s'", index_dir)
    logger.info("=" * 60)
    index = build_index(embeddings)
    save_index(index, chunks, index_dir, embeddings=embeddings)

    # ── Step 5: Build & save BM25 index ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 / 5 — Building BM25 keyword index")
    logger.info("=" * 60)
    bm25 = BM25Index(chunks)
    bm25.save(index_dir)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info("✅  Index built successfully in %.1f seconds", elapsed)
    logger.info("   Documents : %d", len(documents))
    logger.info("   Chunks    : %d", len(chunks))
    logger.info("   Qdrant    : %d vectors (dim=%d)", index.ntotal, embeddings.shape[1])
    logger.info("   BM25      : %d documents", len(chunks))
    logger.info("   Index dir : %s", Path(index_dir).resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
