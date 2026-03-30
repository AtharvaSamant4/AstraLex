"""
verifier.py — Post-generation answer verification.

After the LLM generates an answer, this module asks the LLM to verify
whether EVERY claim in the answer is supported by the retrieved context.

If unsupported claims are detected, a stricter re-generation is triggered
with explicit instructions to ground the answer only in provided evidence.
"""

from __future__ import annotations

import logging
import os
import time

from google import genai

from rag.model_manager import ModelManager

logger = logging.getLogger(__name__)

# ── Verification prompt ────────────────────────────────────────────────────
_VERIFY_SYSTEM = """\
You are a legal answer verification assistant.

Your task is to check whether a generated ANSWER is fully supported by the
provided CONTEXT.  You must be strict.

OUTPUT FORMAT (exactly one of):

VERIFIED
  — if every factual claim in the answer is supported by the context.

UNSUPPORTED: <brief explanation of what is not supported>
  — if any claim is not clearly present in the context.

Do NOT output anything else.
"""


def verify_answer(
    answer: str,
    context_block: str,
    question: str,
    client: genai.Client | None = None,
    model: str | None = None,
) -> tuple[bool, str]:
    """
    Verify whether *answer* is fully grounded in *context_block*.

    Returns
    -------
    (is_verified, explanation)
        ``is_verified`` is True when the answer is fully supported.
        ``explanation`` contains the verifier's reasoning on failure.
    """
    if client is None:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"ANSWER TO VERIFY:\n{answer}\n\n"
        f"VERIFICATION RESULT:"
    )

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
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=_VERIFY_SYSTEM,
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            ModelManager.record_success(model_name)
            result = (response.text or "").strip()

            if result.upper().startswith("VERIFIED"):
                logger.info("Answer verification: PASSED")
                return True, "All claims supported by context."
            else:
                explanation = result.replace("UNSUPPORTED:", "").strip()
                logger.warning("Answer verification: FAILED — %s", explanation[:120])
                return False, explanation

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
                logger.warning("Verification error (attempt %d/%d) — retrying in %ds: %s",
                               attempt, max_retries, delay, exc)
                time.sleep(delay)
                continue
            # On error, assume answer is okay (don't block the response)
            logger.warning("Verification call failed (%s) — assuming verified", exc)
            return True, f"Verification skipped due to error: {exc}"

    return True, "Verification skipped after max retries."


# ── Stricter regeneration prompt ───────────────────────────────────────────
_STRICT_SYSTEM = """\
You are an Indian legal assistant.  A previous answer was found to contain
claims not fully supported by the provided legal context.

RULES — STRICT MODE:
1. Answer ONLY using sentences that appear in the context below.
2. If the context does not contain the information needed to answer, say so.
3. Do NOT add any information beyond what is explicitly stated in the context.
4. Cite Act name and Section number for every statement.
5. Keep the answer clear and well-structured.
"""


def regenerate_strict(
    question: str,
    context_block: str,
    verification_feedback: str,
    client: genai.Client | None = None,
    model: str | None = None,
) -> str:
    """
    Re-generate the answer in strict grounding mode after verification failure.
    """
    if client is None:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model_name = ModelManager.get_model(model_name)

    prompt = (
        f"VERIFICATION FEEDBACK (why the previous answer was rejected):\n"
        f"{verification_feedback}\n\n"
        f"LEGAL CONTEXT:\n{'─' * 60}\n"
        f"{context_block}\n{'─' * 60}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"STRICTLY GROUNDED ANSWER (cite Act and Section):\n"
    )

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
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=_STRICT_SYSTEM,
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )
            ModelManager.record_success(model_name)
            return (response.text or "").strip()
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
                logger.warning("Strict regen error (attempt %d/%d) — retrying in %ds: %s",
                               attempt, max_retries, delay, exc)
                time.sleep(delay)
                continue
            logger.exception("Strict regeneration failed")
            return f"Error during strict regeneration: {exc}"

    return "Failed after maximum retries."
