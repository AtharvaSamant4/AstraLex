"""
query_rewriter.py — LLM-based query understanding and rewriting.

Before retrieval, the raw user question is transformed into an optimized
search query.  This handles:
  • vague / conversational phrasing → precise legal retrieval query
  • abbreviation expansion (IPC, CrPC, HMA, etc.)
  • multi-turn follow-up contextualization using conversation history
"""

from __future__ import annotations

import logging
import os
import time

from google import genai

from rag.model_manager import ModelManager

logger = logging.getLogger(__name__)

# ── Prompt for query rewriting ─────────────────────────────────────────────
_REWRITE_SYSTEM = """\
You are a query rewriting assistant for an Indian legal retrieval system.

Your job is to transform the user's question into an OPTIMIZED SEARCH QUERY
that will work best for retrieving relevant legal sections from a database
of Indian laws (IPC, CrPC, Constitution, Hindu Marriage Act, Special Marriage
Act, Dowry Prohibition Act, Domestic Violence Act).

RULES:
1. Output ONLY the rewritten query — no explanations, no preamble.
2. Expand ALL abbreviations (IPC → Indian Penal Code, CrPC → Code of Criminal
   Procedure, HMA → Hindu Marriage Act, SMA → Special Marriage Act, etc.).
3. Include specific legal terms and section references when implied.
4. If the user's query is already precise, return it mostly unchanged.
5. If conversation history is provided and the user asks a vague follow-up
   (e.g. "what about the punishment?"), incorporate context from previous
   turns to make the query self-contained.
6. Keep the output concise (1–3 sentences max).
"""


def rewrite_query(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
    client: genai.Client | None = None,
    model: str | None = None,
) -> str:
    """
    Rewrite a user question into an optimized retrieval query.

    Parameters
    ----------
    question : str
        Raw user question.
    conversation_history : list[dict] | None
        Previous turns as ``[{"role": "user"|"assistant", "text": ...}, ...]``
    client : genai.Client | None
        Reuse an existing Gemini client, or create one from env.
    model : str | None
        Gemini model name override.

    Returns
    -------
    str
        The rewritten query (falls back to original on error).
    """
    if client is None:
        api_key = os.getenv("GEMINI_API_KEY", "")
        client = genai.Client(api_key=api_key)

    model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model_name = ModelManager.get_model(model_name)

    # Build the user message with optional history
    parts: list[str] = []
    if conversation_history:
        parts.append("CONVERSATION HISTORY:")
        for turn in conversation_history[-6:]:  # last 6 turns max
            role = turn.get("role", "user").upper()
            parts.append(f"  {role}: {turn.get('text', '')}")
        parts.append("")

    parts.append(f"USER QUESTION:\n{question}")
    parts.append("\nREWRITTEN SEARCH QUERY:")

    user_prompt = "\n".join(parts)

    max_retries = 4
    base_delay = 2.0

    attempt = 0
    prev_key_idx = ModelManager.active_key_index()
    max_total = max_retries + ModelManager.total_models() * ModelManager.total_keys()
    for _ in range(max_total):
        # Recreate client if ModelManager switched API keys
        cur_key_idx = ModelManager.active_key_index()
        if cur_key_idx != prev_key_idx:
            client = ModelManager.get_client_for_active_key()
            prev_key_idx = cur_key_idx

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=_REWRITE_SYSTEM,
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            ModelManager.record_success(model_name)
            rewritten = (response.text or "").strip()
            if not rewritten:
                logger.warning("Empty rewrite — using original query")
                return question

            logger.info("Query rewritten: '%s' → '%s'", question[:60], rewritten[:80])
            return rewritten

        except Exception as exc:
            if ModelManager.is_quota_error(exc):
                ModelManager.mark_exhausted(model_name)
                model_name = ModelManager.get_model(model_name)
                continue  # rotate — does NOT count as a retry
            if ModelManager.is_model_incompatible(exc):
                ModelManager.mark_exhausted(model_name)
                logger.warning("Model %s incompatible — rotating", model_name)
                model_name = ModelManager.get_model(model_name)
                continue  # rotate — does NOT count as a retry
            if ModelManager.is_503_error(exc):
                rotated = ModelManager.record_503(model_name)
                if rotated:
                    model_name = ModelManager.get_model(model_name)
                    continue  # auto-rotated — does NOT count as a retry
            attempt += 1
            retryable = ModelManager.is_retryable(exc)
            if retryable and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning("Query rewrite error (attempt %d/%d) — retrying in %ds: %s",
                               attempt, max_retries, delay, exc)
                time.sleep(delay)
                continue
            # Non-retryable or exhausted retries — fall back gracefully
            logger.warning("Query rewrite failed (%s) — using original", exc)
            return question

    return question
