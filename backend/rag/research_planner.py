"""
research_planner.py — LLM-driven research planning for complex questions.

When a legal question arrives, this module asks the LLM to produce a
structured research plan BEFORE any retrieval begins.  The plan:

  • analyses the question's core concepts
  • identifies the relevant areas of law
  • breaks the question into discrete research tasks
  • flags whether multi-hop reasoning is needed

This enables the agentic loop to retrieve evidence systematically rather
than doing a single broad search.
"""

from __future__ import annotations

import json
import logging
import os
import time

from google import genai

from rag.model_manager import ModelManager

logger = logging.getLogger(__name__)

# ── Research-plan prompt ───────────────────────────────────────────────────

_PLAN_SYSTEM = """\
You are a senior Indian legal research planner.

Given a user's legal question, produce a JSON research plan that will guide
an autonomous retrieval system.

OUTPUT — strict JSON, nothing else:
{
  "analysis": "<1-2 sentence restatement of what the user wants to know>",
  "concepts": ["<key legal concept 1>", "<key legal concept 2>", ...],
  "relevant_acts": ["<act name 1>", ...],
  "research_tasks": [
    {
      "id": 1,
      "description": "<what to search for>",
      "search_query": "<optimised retrieval query>",
      "reason": "<why this sub-task is needed>"
    }
  ],
  "requires_multi_hop": true/false,
  "complexity": "simple" | "moderate" | "complex"
}

RULES:
1. Generate 1-5 research tasks depending on complexity.
2. Each search_query must be a concrete, self-contained retrieval query
   suitable for both keyword AND semantic search.
3. Expand all abbreviations (IPC → Indian Penal Code, etc.).
4. For simple questions (single section lookup), 1-2 tasks suffice.
5. For complex questions (comparisons, multi-act, scenario-based), use 3-5 tasks.
6. Always include the user's original intent in the analysis.
7. Output valid JSON only — no markdown fences, no commentary.
"""


def generate_research_plan(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
    client: genai.Client | None = None,
    model: str | None = None,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> dict:
    """
    Ask the LLM to produce a structured research plan for *question*.

    Returns a dict with keys: analysis, concepts, relevant_acts,
    research_tasks, requires_multi_hop, complexity.

    Falls back to a minimal single-task plan on error.
    """
    if client is None:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    model_name = ModelManager.get_model(model_name)

    # Build prompt
    parts: list[str] = []
    if conversation_history:
        parts.append("CONVERSATION HISTORY:")
        for turn in conversation_history[-6:]:
            role = turn.get("role", "user").upper()
            parts.append(f"  {role}: {turn.get('text', '')}")
        parts.append("")
    parts.append(f"USER QUESTION:\n{question}")

    user_prompt = "\n".join(parts)

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
                    system_instruction=_PLAN_SYSTEM,
                    temperature=0.1,
                    max_output_tokens=1024,
                ),
            )
            raw = (response.text or "").strip()

            # Strip markdown fences if the model wrapped its output
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()

            plan = json.loads(raw)

            # Validate minimal structure
            if "research_tasks" not in plan or not plan["research_tasks"]:
                raise ValueError("Plan has no research_tasks")

            ModelManager.record_success(model_name)
            logger.info(
                "Research plan: %d tasks, complexity=%s, multi_hop=%s",
                len(plan["research_tasks"]),
                plan.get("complexity", "?"),
                plan.get("requires_multi_hop", False),
            )
            return plan

        except (json.JSONDecodeError, ValueError, KeyError) as parse_err:
            attempt += 1
            logger.warning("Plan parse error (attempt %d): %s", attempt, parse_err)
            # Fall through to retry
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
                logger.warning("Planner error (attempt %d/%d) — retrying in %ds: %s",
                               attempt, max_retries, delay, exc)
                time.sleep(delay)
                continue
            logger.exception("Research planner failed")
            break  # exit loop — fall through to fallback

    # ── Fallback: minimal plan ─────────────────────────────────────────────
    logger.warning("Falling back to minimal research plan for: %s", question[:80])
    return {
        "analysis": question,
        "concepts": [],
        "relevant_acts": [],
        "research_tasks": [
            {
                "id": 1,
                "description": "General search for the question",
                "search_query": question,
                "reason": "Fallback — planner unavailable",
            }
        ],
        "requires_multi_hop": False,
        "complexity": "simple",
    }
