"""
model_manager.py — Shared model rotation for Gemini API quota management.

Maintains a fallback list of Gemini models.  When any module in the
pipeline encounters a daily quota error (``RESOURCE_EXHAUSTED`` +
``free_tier_requests``), it marks that model as exhausted.  Subsequent
calls to ``get_model()`` will return the next available model.

Also tracks consecutive 503/UNAVAILABLE errors per model.  After
``_503_THRESHOLD`` consecutive 503s on a single model, the model is
automatically rotated out (marked exhausted) to avoid multi-minute
stalls from exponential back-off.

Supports **multiple API keys**.  Call ``init_keys([key1, key2, ...])``
to register them.  When all models exhaust on the current key, the
system auto-switches to the next key, resets model exhaustion, and
continues.  With 2 keys × 9 models × 20 RPD = **360 requests/day**.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

# Ordered by preference: fast models first, then pro.
# Note: gemma-3-27b-it excluded — does not support system_instruction.
# "latest" aliases removed — they share quota with the models they alias.
# gemini-1.5-* removed — deprecated / 404 NOT_FOUND on current API.
_FALLBACK_MODELS: list[str] = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-001",
    "gemini-3.1-flash-lite-preview",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
]

# After this many consecutive 503/UNAVAILABLE errors on a single model,
# mark it exhausted and rotate to the next one.
_503_THRESHOLD = 2


class ModelManager:
    """
    Thread-safe singleton that tracks exhausted Gemini models and
    rotates to fallbacks automatically.

    Usage::

        model = ModelManager.get_model()          # best available
        model = ModelManager.get_model("gemini-2.5-flash")  # preferred
        ModelManager.mark_exhausted("gemini-2.5-flash")     # on quota err
        model = ModelManager.get_model()          # returns next in line
    """

    _exhausted: set[str] = set()
    _503_counts: dict[str, int] = defaultdict(int)
    _lock = threading.Lock()

    # ── Multi-key support ──────────────────────────────────────────────
    _api_keys: list[str] = []
    _current_key_idx: int = 0

    # ── Public API ─────────────────────────────────────────────────────

    @classmethod
    def init_keys(cls, keys: list[str]) -> None:
        """
        Register one or more API keys for rotation.

        Call once at startup (e.g. from pipeline ``__init__``).
        Keys are tried in order; when all models exhaust on key *N*,
        the system switches to key *N+1*.
        """
        with cls._lock:
            cls._api_keys = [k for k in keys if k]
            cls._current_key_idx = 0
            logger.info("Registered %d API key(s) for rotation", len(cls._api_keys))

    @classmethod
    def get_active_key(cls) -> str | None:
        """Return the currently-active API key, or *None* if none registered."""
        with cls._lock:
            if not cls._api_keys:
                return None
            return cls._api_keys[cls._current_key_idx]

    @classmethod
    def active_key_index(cls) -> int:
        """Return the 0-based index of the current API key."""
        return cls._current_key_idx

    @classmethod
    def total_keys(cls) -> int:
        """Return the number of registered API keys."""
        return len(cls._api_keys)

    @classmethod
    def get_model(cls, preferred: str | None = None) -> str:
        """
        Return the best non-exhausted model.

        If *preferred* is given and not exhausted, use it.
        Otherwise walk through the fallback list.
        If all models are exhausted, clear state and start over.
        """
        with cls._lock:
            if preferred and preferred not in cls._exhausted:
                return preferred

            for m in _FALLBACK_MODELS:
                if m not in cls._exhausted:
                    if preferred and m != preferred:
                        logger.info(
                            "Model rotation: %s exhausted → using %s",
                            preferred, m,
                        )
                    return m

            # All exhausted — try next API key if available
            if cls._api_keys and cls._current_key_idx < len(cls._api_keys) - 1:
                cls._current_key_idx += 1
                cls._exhausted.clear()
                cls._503_counts.clear()
                logger.warning(
                    "All %d models exhausted on key #%d — switching to key #%d",
                    len(_FALLBACK_MODELS),
                    cls._current_key_idx,        # already incremented
                    cls._current_key_idx + 1,    # 1-based for log
                )
                return _FALLBACK_MODELS[0]

            # No more keys — clear and retry (last resort)
            logger.warning(
                "All %d models exhausted on all %d key(s) — resetting rotation",
                len(cls._exhausted), max(len(cls._api_keys), 1),
            )
            cls._exhausted.clear()
            cls._503_counts.clear()
            # Reset to first key for a fresh cycle
            cls._current_key_idx = 0
            return _FALLBACK_MODELS[0]

    @classmethod
    def mark_exhausted(cls, model: str) -> None:
        """Mark a model as exhausted (daily quota hit)."""
        with cls._lock:
            if model not in cls._exhausted:
                cls._exhausted.add(model)
                cls._503_counts.pop(model, None)
                remaining = [m for m in _FALLBACK_MODELS if m not in cls._exhausted]
                logger.warning(
                    "Model %s exhausted (daily quota). %d models remain: %s",
                    model, len(remaining),
                    ", ".join(remaining[:3]) + ("…" if len(remaining) > 3 else ""),
                )

    @classmethod
    def record_503(cls, model: str) -> bool:
        """
        Record a 503/UNAVAILABLE error for *model*.

        Returns True if the model was auto-rotated (consecutive count
        reached ``_503_THRESHOLD``), False otherwise.
        """
        with cls._lock:
            cls._503_counts[model] += 1
            count = cls._503_counts[model]
            if count >= _503_THRESHOLD:
                cls._exhausted.add(model)
                cls._503_counts.pop(model, None)
                remaining = [m for m in _FALLBACK_MODELS
                             if m not in cls._exhausted]
                logger.warning(
                    "Model %s hit %d consecutive 503s — auto-rotated. "
                    "%d models remain: %s",
                    model, count, len(remaining),
                    ", ".join(remaining[:3]) + ("…" if len(remaining) > 3 else ""),
                )
                return True
            logger.info("Model %s got 503 (%d/%d before rotation)",
                        model, count, _503_THRESHOLD)
            return False

    @classmethod
    def record_success(cls, model: str) -> None:
        """Reset 503 counter on a successful call."""
        with cls._lock:
            cls._503_counts.pop(model, None)

    @classmethod
    def is_503_error(cls, exc: Exception) -> bool:
        """Return True if the exception is a 503/UNAVAILABLE error."""
        msg = str(exc)
        return "503" in msg or "UNAVAILABLE" in msg

    @classmethod
    def is_quota_error(cls, exc: Exception) -> bool:
        """
        Return True if the exception indicates a daily quota exhaustion
        (not a transient rate-limit).
        """
        msg = str(exc)
        return (
            "RESOURCE_EXHAUSTED" in msg
            and "free_tier_requests" in msg
        )

    @classmethod
    def is_model_incompatible(cls, exc: Exception) -> bool:
        """
        Return True if the model doesn't support a required feature
        (e.g. system_instruction not supported for gemma models) or
        if the model is not found (404 / deprecated).
        """
        msg = str(exc)
        return (
            ("INVALID_ARGUMENT" in msg
             and ("not enabled" in msg or "not supported" in msg))
            or ("NOT_FOUND" in msg and "not found" in msg.lower())
        )

    @classmethod
    def is_retryable(cls, exc: Exception) -> bool:
        """Return True for transient / retryable errors."""
        msg = str(exc)
        return any(tok in msg for tok in (
            "429", "RESOURCE_EXHAUSTED",
            "503", "UNAVAILABLE",
            "500", "INTERNAL",
            "overloaded", "high demand",
        ))

    @classmethod
    def reset(cls) -> None:
        """Reset exhaustion state (e.g. for testing)."""
        with cls._lock:
            cls._exhausted.clear()
            cls._503_counts.clear()
            cls._current_key_idx = 0

    @classmethod
    def get_client_for_active_key(cls):
        """
        Return a ``genai.Client`` wired to the currently-active API key.

        Sub-modules that receive a *client* parameter should call this
        after a key rotation so they don't keep using a stale key.
        """
        import google.genai as genai

        key = cls.get_active_key()
        if not key:
            raise RuntimeError("No API keys registered with ModelManager")
        return genai.Client(api_key=key)

    @classmethod
    def exhausted_count(cls) -> int:
        """Number of currently exhausted models."""
        return len(cls._exhausted)

    @classmethod
    def total_models(cls) -> int:
        """Total number of available fallback models."""
        return len(_FALLBACK_MODELS)
