"""
context_builder.py — Smart context construction from reranked chunks.

Instead of blindly concatenating retrieved text, this module:
  1. Deduplicates overlapping content (common with chunk overlap).
  2. Removes chunks whose text is a near-subset of another selected chunk.
  3. Structures each piece with source metadata for the prompt.
  4. Optionally performs LLM-based context compression to extract only the
     sentences relevant to the user's question.
"""

from __future__ import annotations

import logging
import os
from difflib import SequenceMatcher

from google import genai

from rag.chunker import Chunk

logger = logging.getLogger(__name__)

# ── Similarity threshold for deduplication ─────────────────────────────────
_DEDUP_THRESHOLD = 0.80  # ≥80 % text overlap → consider duplicate


def _text_similarity(a: str, b: str) -> float:
    """Quick ratio similarity between two strings."""
    return SequenceMatcher(None, a, b).quick_ratio()


def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """
    Remove chunks that are near-duplicates of an already-selected chunk.

    Uses ``difflib.SequenceMatcher.quick_ratio`` for speed.
    """
    if len(chunks) <= 1:
        return chunks

    selected: list[Chunk] = [chunks[0]]
    for candidate in chunks[1:]:
        is_dup = any(
            _text_similarity(candidate["text"], sel["text"]) >= _DEDUP_THRESHOLD
            for sel in selected
        )
        if not is_dup:
            selected.append(candidate)
        else:
            logger.debug("Deduplicated chunk %s", candidate.get("chunk_id", "?"))

    logger.info("Deduplication: %d → %d chunks", len(chunks), len(selected))
    return selected


def build_context_block(
    chunks: list[Chunk],
    scores: list[float] | None = None,
) -> str:
    """
    Build a numbered context block string ready for insertion in a prompt.

    Each chunk is prefixed with its source metadata.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = f"{chunk['act']} — {chunk['section']}: {chunk['title']}"
        score_str = f" (relevance {scores[i-1]:.3f})" if scores else ""
        parts.append(
            f"[{i}] SOURCE: {source}{score_str}\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(parts)


# ── LLM-based context compression (optional, costs one API call) ───────────

_COMPRESS_SYSTEM = """\
You are a legal context compression assistant.

Given a set of legal text passages and a user question, extract ONLY the
sentences and provisions that are directly relevant to answering the question.

RULES:
1. Preserve exact legal wording — do not paraphrase legal provisions.
2. Keep section numbers and act names.
3. Remove unrelated sentences, procedural boilerplate, and repetition.
4. Output the compressed context as numbered items matching the original
   source labels [1], [2], etc.  Drop items that are entirely irrelevant.
5. If a passage is fully relevant, include it unchanged.
"""


def compress_context(
    context_block: str,
    question: str,
    client: genai.Client | None = None,
    model: str | None = None,
) -> str:
    """
    Use the LLM to compress *context_block* by removing irrelevant sentences.

    Falls back to the uncompressed block on error.
    """
    if client is None:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

    model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT TO COMPRESS:\n{context_block}\n\n"
        f"COMPRESSED CONTEXT (relevant parts only):"
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=_COMPRESS_SYSTEM,
                temperature=0.0,
                max_output_tokens=1536,
            ),
        )
        compressed = (response.text or "").strip()
        if not compressed:
            logger.warning("Empty compression result — using original context")
            return context_block

        orig_len = len(context_block)
        comp_len = len(compressed)
        logger.info(
            "Context compressed: %d → %d chars (%.0f%% reduction)",
            orig_len, comp_len, (1 - comp_len / orig_len) * 100 if orig_len else 0,
        )
        return compressed

    except Exception as exc:
        logger.warning("Context compression failed (%s) — using original", exc)
        return context_block
