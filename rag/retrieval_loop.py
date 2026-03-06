"""
retrieval_loop.py — Autonomous iterative retrieval controller.

Instead of running a single retrieval pass, this module implements an
agentic loop:

    1. Execute all research tasks (initial retrieval)
    2. Analyse collected evidence for gaps
    3. Generate follow-up queries for missing information
    4. Retrieve additional evidence
    5. Repeat until sufficient or max iterations reached

The loop uses the LLM as a "gap analyst" to decide whether more
evidence is needed and what to search for next.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np
from google import genai

from rag.bm25_search import BM25Index
from rag.chunker import Chunk
from rag.context_builder import deduplicate_chunks
from rag.embedder import embed_query
from rag.evidence_graph import EvidenceGraph, build_evidence_graph
from rag.model_manager import ModelManager
from rag.reranker import rerank
from rag.vectordb import search_index

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────

@dataclass
class RetrievalLoopConfig:
    """Knobs for the iterative retrieval loop."""
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 8          # per-task top-k after reranking
    max_iterations: int = 3        # max retrieval rounds (1 = no follow-up)
    max_total_chunks: int = 40     # stop collecting beyond this
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gemini_model: str = "gemini-2.5-flash"
    max_retries: int = 3
    base_delay: float = 2.0


# ── Result ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Everything the retrieval loop produces."""
    evidence_graph: EvidenceGraph
    all_chunks: list[Chunk]
    final_chunks: list[Chunk]
    iteration_count: int
    follow_up_queries: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


# ── RRF fusion (local copy to avoid circular import) ───────────────────────

def _rrf_fuse(
    dense_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    rrf: dict[int, float] = {}
    for rank, (idx, _) in enumerate(dense_results):
        rrf[idx] = rrf.get(idx, 0) + 1.0 / (k + rank + 1)
    for rank, (idx, _) in enumerate(bm25_results):
        rrf[idx] = rrf.get(idx, 0) + 1.0 / (k + rank + 1)
    return sorted(rrf.items(), key=lambda x: x[1], reverse=True)


# ── Gap analysis prompt ───────────────────────────────────────────────────

_GAP_SYSTEM = """\
You are a legal research gap analyst. Given a question, a research plan,
and the evidence collected so far, determine whether MORE evidence is needed.

OUTPUT — strict JSON, nothing else:
{
  "sufficient": true/false,
  "reason": "<why evidence is or isn't sufficient>",
  "follow_up_queries": ["<query 1>", "<query 2>"]
}

RULES:
1. If the evidence already covers all aspects of the question → sufficient=true,
   follow_up_queries=[].
2. If key information is missing → sufficient=false, provide 1-3 follow-up queries.
3. Follow-up queries must be concrete retrieval queries.
4. Output valid JSON only — no markdown fences, no commentary.
"""


def _analyse_gaps(
    question: str,
    plan: dict,
    evidence_summary: str,
    client: genai.Client,
    model: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> dict:
    """Ask the LLM whether more evidence is needed."""
    model = ModelManager.get_model(model)
    prompt = (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"RESEARCH PLAN:\n{json.dumps(plan, indent=2)}\n\n"
        f"EVIDENCE COLLECTED SO FAR:\n{evidence_summary[:3000]}\n\n"
        f"GAP ANALYSIS:"
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=_GAP_SYSTEM,
                    temperature=0.0,
                    max_output_tokens=512,
                ),
            )
            raw = (response.text or "").strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw.rsplit("```", 1)[0]
            raw = raw.strip()
            ModelManager.record_success(model)
            return json.loads(raw)
        except Exception as exc:
            if ModelManager.is_quota_error(exc):
                ModelManager.mark_exhausted(model)
                model = ModelManager.get_model(model)
                continue  # retry with next model
            if ModelManager.is_model_incompatible(exc):
                ModelManager.mark_exhausted(model)
                logger.warning("Model %s incompatible — rotating", model)
                model = ModelManager.get_model(model)
                continue
            if ModelManager.is_503_error(exc):
                rotated = ModelManager.record_503(model)
                if rotated:
                    model = ModelManager.get_model(model)
                    continue
            retryable = ModelManager.is_retryable(exc)
            if retryable and attempt < max_retries:
                time.sleep(base_delay * (2 ** (attempt - 1)))
                continue
            logger.warning("Gap analysis failed (%s) — assuming sufficient", exc)
            return {"sufficient": True, "reason": "Gap analysis error", "follow_up_queries": []}

    return {"sufficient": True, "reason": "Max retries", "follow_up_queries": []}


# ── Main loop ──────────────────────────────────────────────────────────────

def run_retrieval_loop(
    question: str,
    plan: dict,
    faiss_index,
    chunks: list[Chunk],
    bm25: BM25Index,
    client: genai.Client,
    config: RetrievalLoopConfig | None = None,
) -> RetrievalResult:
    """
    Execute the iterative retrieval loop.

    1. Run all research tasks from the plan (initial retrieval).
    2. Build evidence graph.
    3. Ask gap analyst if more evidence is needed.
    4. If yes, run follow-up retrieval.
    5. Repeat up to ``config.max_iterations`` times.
    """
    config = config or RetrievalLoopConfig()
    timings: dict[str, float] = {}
    all_collected: list[Chunk] = []
    all_scores: list[float] = []
    all_task_ids: list[int | None] = []
    follow_up_queries: list[str] = []
    seen_chunk_ids: set[str] = set()

    def _hybrid_retrieve(query: str, task_id: int | None = None) -> list[tuple[Chunk, float]]:
        """Single hybrid retrieval pass for one query."""
        qvec = embed_query(query, model_name=config.embedding_model)
        d_scores, d_indices = search_index(faiss_index, qvec, top_k=config.dense_top_k)
        dense = [
            (int(idx), float(sc))
            for sc, idx in zip(d_scores[0], d_indices[0])
            if idx != -1
        ]
        bm25_res = bm25.search(query, top_k=config.bm25_top_k)
        fused = _rrf_fuse(dense, bm25_res, k=config.rrf_k)

        candidates = [chunks[idx] for idx, _ in fused[:40]]
        if not candidates:
            return []

        reranked = rerank(
            query=query,
            chunks=candidates,
            model_name=config.reranker_model,
            top_k=config.rerank_top_k,
        )
        return reranked

    def _collect(results: list[tuple[Chunk, float]], task_id: int | None) -> int:
        """Add new chunks to the pool, dedup by chunk_id. Returns count added."""
        added = 0
        for chunk, score in results:
            cid = chunk.get("chunk_id", "")
            if cid and cid in seen_chunk_ids:
                continue
            seen_chunk_ids.add(cid)
            all_collected.append(chunk)
            all_scores.append(score)
            all_task_ids.append(task_id)
            added += 1
        return added

    # ── Iteration 1: execute all research tasks ────────────────────────────
    t0 = time.perf_counter()
    tasks = plan.get("research_tasks", [{"search_query": question}])
    for task in tasks:
        query = task.get("search_query", question)
        tid = task.get("id")
        results = _hybrid_retrieve(query, tid)
        _collect(results, tid)
        logger.info("Task %s: retrieved %d chunks for '%s'",
                     tid, len(results), query[:60])

        # Safety cap
        if len(all_collected) >= config.max_total_chunks:
            break

    timings["initial_retrieval"] = time.perf_counter() - t0
    iteration = 1

    # ── Iterations 2+: gap analysis → follow-up retrieval ──────────────────
    while iteration < config.max_iterations and len(all_collected) < config.max_total_chunks:
        t0 = time.perf_counter()

        # Build a quick summary of what we have so far
        evidence_summary_parts = []
        for i, c in enumerate(all_collected[:20]):
            evidence_summary_parts.append(
                f"[{i+1}] {c['act']} — {c['section']}: {c['title']}"
            )
        evidence_summary = "\n".join(evidence_summary_parts)

        gap = _analyse_gaps(
            question, plan, evidence_summary,
            client, config.gemini_model,
            config.max_retries, config.base_delay,
        )

        if gap.get("sufficient", True):
            logger.info("Gap analyst: evidence is sufficient after %d iteration(s)", iteration)
            timings[f"gap_analysis_{iteration}"] = time.perf_counter() - t0
            break

        follow_ups = gap.get("follow_up_queries", [])
        if not follow_ups:
            logger.info("Gap analyst: no follow-up queries generated")
            timings[f"gap_analysis_{iteration}"] = time.perf_counter() - t0
            break

        follow_up_queries.extend(follow_ups)
        logger.info("Gap analyst: %d follow-up queries — %s",
                     len(follow_ups), follow_ups)

        timings[f"gap_analysis_{iteration}"] = time.perf_counter() - t0

        # Execute follow-up retrievals
        t0 = time.perf_counter()
        for fq in follow_ups:
            results = _hybrid_retrieve(fq)
            added = _collect(results, None)
            logger.info("Follow-up: +%d new chunks for '%s'", added, fq[:60])
            if len(all_collected) >= config.max_total_chunks:
                break

        timings[f"followup_retrieval_{iteration}"] = time.perf_counter() - t0
        iteration += 1

    # ── Build evidence graph ───────────────────────────────────────────────
    t0 = time.perf_counter()
    evidence_graph = build_evidence_graph(all_collected, all_scores, all_task_ids)
    timings["graph_build"] = time.perf_counter() - t0

    # ── Final dedup + top selection ────────────────────────────────────────
    top_nodes = evidence_graph.top_nodes(k=config.rerank_top_k * 2)
    final_chunks = deduplicate_chunks([n.chunk for n in top_nodes if n.chunk])

    return RetrievalResult(
        evidence_graph=evidence_graph,
        all_chunks=all_collected,
        final_chunks=final_chunks,
        iteration_count=iteration,
        follow_up_queries=follow_up_queries,
        timings=timings,
    )
