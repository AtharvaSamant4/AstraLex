"""
pipeline.py — Deep-Research Agentic RAG Pipeline with Adaptive Routing.

Routes every legal question through one of three tiers based on
complexity, to minimize API calls while maximising answer quality:

    FAST   (1 LLM call)  — direct section lookups, clear single-topic Qs
    STANDARD (2 LLM calls) — typical legal questions needing rewrite + gen
    DEEP   (3 LLM calls) — complex multi-hop / comparison / scenario Qs

Tier assignment is done with a zero-cost regex heuristic (no LLM call).

Non-legal queries (greetings, off-topic) get a single conversational call.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from google import genai

from rag.bm25_search import BM25Index
from rag.chunker import Chunk
from rag.context_builder import build_context_block, deduplicate_chunks
from rag.embedder import embed_query
from rag.evidence_graph import build_evidence_graph
from rag.intent_classifier import classify_intent
from rag.prompt_template import (
    DEEP_RESEARCH_SYSTEM,
    SYSTEM_INSTRUCTION,
    build_advanced_prompt,
    build_deep_research_prompt,
)
from rag.model_manager import ModelManager
from rag.query_rewriter import rewrite_query
from rag.reranker import rerank
from rag.research_planner import generate_research_plan
from rag.retrieval_loop import RetrievalLoopConfig, run_retrieval_loop
from rag.vectordb import load_index, search_index
from rag.verifier import regenerate_strict, verify_answer

logger = logging.getLogger(__name__)


# ── Query complexity heuristic (zero LLM cost) ────────────────────────────

_SECTION_LOOKUP_RE = re.compile(
    r"""(?ix)
    ^(?:what\s+(?:is|does|are)|explain|describe|tell\s+me\s+about|state)
    [^?]*?
    (?:section|article|sec\.?|art\.?)
    \s*\d+[A-Za-z]?
    """,
)

_DIRECT_LOOKUP_RE = re.compile(
    r"""(?ix)
    ^(?:what\s+(?:is|does|are)|explain|describe)
    [^?]*?
    (?:ipc|crpc|constitution|hindu\s+marriage|special\s+marriage
      |dowry|domestic\s+violence)
    [^?]*?
    (?:section|article|sec\.?|art\.?)
    \s*\d+
    """,
)

_MULTI_HOP_RE = re.compile(
    r"""(?ix)
    (?:
        difference\s+between
      | compare|comparison
      | how\s+does\s+.+differ
      | distinguish\s+between
      | versus|vs\.?
      | both\s+.+and
      | including\s+their
      | what\s+legal\s+(?:protections?|provisions?|rights?)\s+exist
      | what\s+happens?\s+(?:legally\s+)?if
      | under\s+(?:indian|criminal)\s+(?:law|criminal\s+law)
    )
    """,
)

_AMBIGUOUS_SHORT_RE = re.compile(
    r"^[\w\s]{2,30}$",  # very short queries like "murder punishment"
)

# Follow-up patterns — short queries that reference a previous turn.
_FOLLOW_UP_RE = re.compile(
    r"""(?ix)
    ^(?:
        what\s+about
      | how\s+about
      | and\s+(?:what|how|if|the|for)
      | what\s+(?:is|are|was|were)\s+(?:that|those|this|its?)
      | which\s+(?:section|article|act|law|provision|one)
      | tell\s+me\s+more
      | explain\s+(?:that|this|it|simply|more|further|again|in\s+(?:simple|easy|detail|one))
      | can\s+you\s+(?:explain|elaborate|clarify|simplify)
      | what\s+(?:does|do)\s+(?:that|this|it)\s+mean
      | what\s+(?:else|more)
      | anything\s+else
      | go\s+on
      | continue
      | more\s+(?:details?|info|information)
      | what'?s?\s+the\s+(?:section|article|provision|penalty|punishment)
      | is\s+(?:that|this|it)
      | and\s+(?:the|its?)\s+(?:punishment|penalty|section|article)
      | (?:in\s+)?(?:simple|easy|plain)\s+(?:words?|language|terms?|way)?
      | (?:make|put)\s+it\s+(?:simple|easy|short)
      | simplify
      | easy\s+please
      | (?:in\s+)?one\s+line
      | (?:in\s+)?short
      | summarize|summarise|sum\s+up|sum\s+it
      | what\s+(?:is|was)\s+(?:this|that)
    )
    """,
)


def _classify_query_tier(question: str) -> str:
    """
    Classify a legal query into a processing tier using regex heuristics.
    No LLM call required.

    Returns one of: "fast", "standard", "deep"
    """
    q = question.strip()

    # Section/article lookups → fast path (1 LLM call)
    if _SECTION_LOOKUP_RE.search(q) or _DIRECT_LOOKUP_RE.search(q):
        return "fast"

    # Multi-hop / comparison / complex reasoning → deep path
    if _MULTI_HOP_RE.search(q):
        return "deep"

    # Very short / ambiguous → standard (needs rewrite but not deep research)
    if _AMBIGUOUS_SHORT_RE.match(q) and len(q.split()) <= 4:
        return "standard"

    # Default: standard for most questions
    return "standard"


# ── RRF fusion ─────────────────────────────────────────────────────────────

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


# ── Pipeline configuration ─────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """All tuneable knobs of the deep-research RAG pipeline."""

    # Retrieval
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5

    # Iterative retrieval (deep tier only)
    max_retrieval_iterations: int = 2
    max_total_chunks: int = 40

    # Verification — OFF by default to conserve API budget
    verify_answer: bool = False

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gemini_model: str = "gemini-2.5-flash"

    # Retry
    max_retries: int = 4
    base_delay: float = 2.0


# ── Pipeline result ────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Everything the pipeline produces."""
    answer: str
    sources: list[str]
    rewritten_query: str
    retrieved_chunks: list[Chunk]
    final_chunks: list[Chunk]
    verified: bool
    verification_note: str
    timings: dict[str, float] = field(default_factory=dict)

    # Deep-research metadata
    research_plan: dict | None = None
    evidence_graph_stats: dict[str, int] | None = None
    retrieval_iterations: int = 1
    follow_up_queries: list[str] = field(default_factory=list)
    complexity: str = "simple"
    tier: str = "standard"


# ── Main pipeline class ───────────────────────────────────────────────────

class RAGPipeline:
    """
    Deep-Research Agentic RAG Pipeline with adaptive 3-tier routing.

    Tier routing (zero-cost regex heuristic):
      FAST     — section lookups  → 1 LLM call (generate only)
      STANDARD — typical legal Qs → 2 LLM calls (rewrite + generate)
      DEEP     — complex / multi-hop → 3 LLM calls (rewrite + plan + generate)

    Verification is OFF by default; enable via config.verify_answer=True.
    """

    def __init__(
        self,
        index_dir: str | Path = "index",
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig(
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            rerank_top_k=int(os.getenv("TOP_K", "5")),
        )
        self.index_dir = Path(index_dir)

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise EnvironmentError("GEMINI_API_KEY is not set.")

        # Collect all API keys (GEMINI_API_KEY, GEMINI_API_KEY_2, …)
        api_keys = [api_key]
        for i in range(2, 20):  # support up to 19 extra keys
            extra = os.getenv(f"GEMINI_API_KEY_{i}", "")
            if extra and extra != "your_gemini_api_key_here":
                api_keys.append(extra)
        ModelManager.init_keys(api_keys)

        self._client = genai.Client(api_key=api_key)
        self._active_key = api_key  # track which key the client uses

        self._vector_index, self._chunks = load_index(self.index_dir)
        self._bm25 = BM25Index.load(self._chunks, self.index_dir)
        self._history: list[dict[str, str]] = []

        logger.info(
            "Pipeline ready — %d chunks, models: embed=%s, rerank=%s, llm=%s",
            len(self._chunks),
            self.config.embedding_model,
            self.config.reranker_model,
            self.config.gemini_model,
        )

    # ── Shared retrieval (no LLM call) ─────────────────────────────────────

    def _hybrid_retrieve(
        self, query: str, top_k: int | None = None,
    ) -> tuple[list[Chunk], list[float]]:
        """
        Run hybrid retrieval (FAISS + BM25 → RRF → rerank).
        Zero LLM calls — only embedding + BM25 + cross-encoder.
        """
        top_k = top_k or self.config.rerank_top_k

        qvec = embed_query(query, model_name=self.config.embedding_model)
        d_scores, d_indices = search_index(
            self._vector_index, qvec, top_k=self.config.dense_top_k,
        )
        dense = [(int(idx), float(sc))
                 for sc, idx in zip(d_scores[0], d_indices[0]) if idx != -1]

        bm25_res = self._bm25.search(query, top_k=self.config.bm25_top_k)
        fused = _rrf_fuse(dense, bm25_res, k=self.config.rrf_k)
        candidates = [self._chunks[idx] for idx, _ in fused[:40]]

        if not candidates:
            return [], []

        reranked = rerank(
            query=query, chunks=candidates,
            model_name=self.config.reranker_model,
            top_k=top_k,
        )
        chunks = deduplicate_chunks([c for c, _ in reranked])
        scores = [sc for _, sc in reranked][:len(chunks)]
        return chunks, scores

    # ── Act alias mapping for fast-tier filtering ─────────────────────────

    _ACT_ALIASES: dict[str, str] = {
        "ipc": "Indian Penal Code (IPC)",
        "indian penal code": "Indian Penal Code (IPC)",
        "crpc": "Code of Criminal Procedure (CrPC)",
        "criminal procedure": "Code of Criminal Procedure (CrPC)",
        "code of criminal procedure": "Code of Criminal Procedure (CrPC)",
        "constitution": "Constitution of India",
        "hindu marriage": "Hindu Marriage Act",
        "special marriage": "Special Marriage Act",
        "dowry": "Dowry Prohibition Act",
        "domestic violence": "Protection of Women from Domestic Violence Act",
    }

    _ACT_MENTION_RE = re.compile(
        r"(?i)\b("
        + "|".join(re.escape(k) for k in sorted(_ACT_ALIASES, key=len, reverse=True))
        + r")\b"
    )

    # ── Follow-up detection ─────────────────────────────────────────────

    def _is_legal_followup(self, question: str) -> bool:
        """Return *True* if *question* looks like a follow-up to a recent
        legal exchange (so it should be routed to the standard pipeline
        rather than treated as off-topic / conversational).

        Two conditions must BOTH be true:
        1. Recent conversation history contains a substantive legal answer
           (not a greeting / "I can help you with…" placeholder).
        2. The current query either matches ``_FOLLOW_UP_RE`` patterns
           (e.g. "what about …", "which section …", "explain simply")
           **or** is very short (≤ 6 words) — typical of contextual
           follow-ups like "and the penalty?" or "attempt?".
        """
        if not self._history:
            return False

        # Look back further (up to 8 entries ≈ 4 turns) to survive
        # quota-failed or empty answers that may have broken the
        # immediately-preceding turn.
        recent = self._history[-8:]
        _CONV_PREFIXES = (
            "I can help you with questions",
            "Hello",
            "Hi!",
            "Goodbye",
            "You're welcome",
            "I'm here to help",
            "Please provide a question",
        )
        _FAILURE_MARKERS = (
            "Failed after maximum retries",
            "Error communicating with Gemini",
            "I couldn't generate",
        )
        has_legal_context = any(
            entry["role"] == "assistant"
            and len(entry["text"]) > 120
            and not any(entry["text"].strip().startswith(p) for p in _CONV_PREFIXES)
            and not any(m in entry["text"] for m in _FAILURE_MARKERS)
            for entry in recent
        )
        if not has_legal_context:
            return False

        q = question.strip()
        if _FOLLOW_UP_RE.search(q):
            return True
        if len(q.split()) <= 6:
            return True
        return False

    def _detect_act(self, question: str) -> str | None:
        """Return the canonical act name if the question mentions one."""
        m = self._ACT_MENTION_RE.search(question)
        if m:
            return self._ACT_ALIASES.get(m.group(1).lower())
        return None

    # ── Direct section/article lookup by metadata ─────────────────────────

    _SECTION_NUM_RE = re.compile(
        r"(?i)\b(?:section|sec\.?|article|art\.?)\s*(\d+[A-Za-z]*)\b"
    )

    def _direct_section_lookup(
        self, question: str, act: str | None = None,
    ) -> list[Chunk]:
        """
        Find chunks whose section metadata exactly matches the section
        number mentioned in the question.  If an act is also specified,
        filter to that act only.  Returns matching chunks (may be empty).
        """
        m = self._SECTION_NUM_RE.search(question)
        if not m:
            return []

        target_num = m.group(1).upper()  # e.g. "300", "498A"
        matches: list[Chunk] = []
        for chunk in self._chunks:
            sec = chunk.get("section", "")
            # section field looks like "Section 300" or "Article 21"
            sec_num = sec.split()[-1].upper() if sec else ""
            if sec_num == target_num:
                if act is None or chunk.get("act") == act:
                    matches.append(chunk)

        if matches:
            logger.info(
                "Direct section lookup: found %d chunk(s) for %s%s",
                len(matches), target_num,
                f" in {act}" if act else "",
            )
        return matches

    # ── FAST tier (1 LLM call) ─────────────────────────────────────────────

    def _run_fast(self, question: str, timings: dict) -> PipelineResult:
        """
        Section lookups and direct queries.
        Uses the original question directly — no rewrite, no plan.
        1 LLM call: generation only.
        """
        t0 = time.perf_counter()
        chunks, scores = self._hybrid_retrieve(question)
        timings["retrieval"] = time.perf_counter() - t0

        # ── Act-aware filtering ────────────────────────────────────────────
        # If the user explicitly mentioned an act (e.g. "IPC Section 300"),
        # filter to only chunks from that act.  If no chunks remain, it means
        # the requested section isn't in our database.
        mentioned_act = self._detect_act(question)
        if mentioned_act and chunks:
            matching = [(c, s) for c, s in zip(chunks, scores)
                        if c["act"] == mentioned_act]
            if matching:
                chunks = [c for c, _ in matching]
                scores = [s for _, s in matching]
            else:
                chunks = []
                scores = []

        # ── Direct section lookup (metadata-based fallback) ───────────────
        # If semantic+BM25 retrieval missed the exact section, try a direct
        # metadata lookup.  This handles cases where BM25/dense search
        # fails to rank the correct section highly enough (e.g. long texts
        # penalised by BM25 length normalisation).
        has_specific_section = bool(self._SECTION_NUM_RE.search(question))
        direct_hits = self._direct_section_lookup(question, mentioned_act)
        if direct_hits:
            # Merge direct hits into retrieved chunks (dedup by text)
            existing_texts = {c["text"] for c in chunks}
            for dh in direct_hits:
                if dh["text"] not in existing_texts:
                    chunks.insert(0, dh)  # prepend for priority
                    scores.insert(0, 1.0)  # high synthetic score
                    existing_texts.add(dh["text"])
        elif has_specific_section and mentioned_act:
            # User asked for a specific section of a specific act (e.g.
            # "IPC Section 999") but neither retrieval nor direct lookup
            # found it.  Return a clear "not found" instead of showing
            # unrelated chunks from the same act.
            sec_match = self._SECTION_NUM_RE.search(question)
            sec_num = sec_match.group(1) if sec_match else "unknown"
            return PipelineResult(
                answer=(
                    f"Section {sec_num} of the {mentioned_act} is not available "
                    f"in our database. Our database covers select sections of "
                    f"this act. Please try asking about a different section, or "
                    f"rephrase your question."
                ),
                sources=[], rewritten_query=question,
                retrieved_chunks=[], final_chunks=[],
                verified=True,
                verification_note="Section not found in database.",
                timings=timings, complexity="simple", tier="fast",
            )

        if not chunks:
            if mentioned_act:
                return PipelineResult(
                    answer=(
                        f"The specific section you asked about is not currently "
                        f"available in our {mentioned_act} database. "
                        f"Our database covers select sections of this act. "
                        f"Please try asking about a different section, or "
                        f"rephrase your question."
                    ),
                    sources=[], rewritten_query=question,
                    retrieved_chunks=[], final_chunks=[],
                    verified=True,
                    verification_note="Section not found in database.",
                    timings=timings, complexity="simple", tier="fast",
                )
            return self._no_docs_result(question, question, "fast")

        t0 = time.perf_counter()
        context = build_context_block(chunks, scores)
        prompt = build_advanced_prompt(question, context)
        answer = self._call_gemini(prompt, SYSTEM_INSTRUCTION)
        timings["generation"] = time.perf_counter() - t0

        sources = self._extract_sources(chunks)
        return PipelineResult(
            answer=answer, sources=sources, rewritten_query=question,
            retrieved_chunks=chunks, final_chunks=chunks,
            verified=True, verification_note="Fast tier — verification skipped.",
            timings=timings, complexity="simple", tier="fast",
        )

    # ── STANDARD tier (2 LLM calls) ───────────────────────────────────────

    def _run_standard(self, question: str, timings: dict) -> PipelineResult:
        """
        Typical legal questions.
        2 LLM calls: query rewrite + generation.
        """
        # Rewrite (1 LLM call)
        t0 = time.perf_counter()
        rewritten = rewrite_query(
            question,
            conversation_history=self._history[-6:] if self._history else None,
            client=self._client, model=self._active_model(),
        )
        timings["query_rewrite"] = time.perf_counter() - t0

        # Retrieval (0 LLM calls)
        t0 = time.perf_counter()
        chunks, scores = self._hybrid_retrieve(rewritten)
        timings["retrieval"] = time.perf_counter() - t0

        if not chunks:
            return self._no_docs_result(question, rewritten, "standard")

        # Build context + generate (1 LLM call)
        t0 = time.perf_counter()
        context = build_context_block(chunks, scores)
        prompt = build_advanced_prompt(question, context, rewritten)
        answer = self._call_gemini(prompt, SYSTEM_INSTRUCTION)
        timings["generation"] = time.perf_counter() - t0

        # Optional verification
        verified, vnote = True, "Verification skipped."
        if self.config.verify_answer and answer:
            t0 = time.perf_counter()
            verified, vnote = verify_answer(
                answer, context, question,
                client=self._client, model=self._active_model(),
            )
            if not verified:
                answer = regenerate_strict(
                    question, context, vnote,
                    client=self._client, model=self._active_model(),
                )
                verified, vnote = True, "Re-generated in strict mode."
            timings["verification"] = time.perf_counter() - t0

        sources = self._extract_sources(chunks)
        return PipelineResult(
            answer=answer, sources=sources, rewritten_query=rewritten,
            retrieved_chunks=chunks, final_chunks=chunks,
            verified=verified, verification_note=vnote,
            timings=timings, complexity="moderate", tier="standard",
        )

    # ── DEEP tier (3 LLM calls) ───────────────────────────────────────────

    def _run_deep(self, question: str, timings: dict) -> PipelineResult:
        """
        Complex multi-hop / comparison / scenario questions.
        3 LLM calls: rewrite + research plan + generation.
        Gap analysis only if plan says requires_multi_hop=True.
        """
        # Rewrite (1 LLM call)
        t0 = time.perf_counter()
        rewritten = rewrite_query(
            question,
            conversation_history=self._history[-6:] if self._history else None,
            client=self._client, model=self._active_model(),
        )
        timings["query_rewrite"] = time.perf_counter() - t0

        # Research plan (1 LLM call)
        t0 = time.perf_counter()
        plan = generate_research_plan(
            question=rewritten,
            conversation_history=self._history[-6:] if self._history else None,
            client=self._client, model=self._active_model(),
        )
        timings["research_planning"] = time.perf_counter() - t0
        complexity = plan.get("complexity", "complex")

        # Iterative retrieval loop — gap analysis uses 0–1 LLM calls
        # Set max_iterations based on plan complexity
        needs_multi_hop = plan.get("requires_multi_hop", False)
        max_iter = self.config.max_retrieval_iterations if needs_multi_hop else 1

        t0 = time.perf_counter()
        loop_config = RetrievalLoopConfig(
            dense_top_k=self.config.dense_top_k,
            bm25_top_k=self.config.bm25_top_k,
            rrf_k=self.config.rrf_k,
            rerank_top_k=self.config.rerank_top_k,
            max_iterations=max_iter,
            max_total_chunks=self.config.max_total_chunks,
            embedding_model=self.config.embedding_model,
            reranker_model=self.config.reranker_model,
            gemini_model=self._active_model(),
            max_retries=self.config.max_retries,
            base_delay=self.config.base_delay,
        )
        retrieval_result = run_retrieval_loop(
            question=rewritten, plan=plan,
            vector_index=self._vector_index, chunks=self._chunks,
            bm25=self._bm25, client=self._client, config=loop_config,
        )
        timings["retrieval_loop"] = time.perf_counter() - t0
        timings.update(retrieval_result.timings)

        evidence_graph = retrieval_result.evidence_graph
        final_chunks = retrieval_result.final_chunks
        graph_stats = evidence_graph.summary_stats()

        if not final_chunks:
            return self._no_docs_result(
                question, rewritten, "deep",
                plan=plan, graph_stats=graph_stats,
                iterations=retrieval_result.iteration_count,
                complexity=complexity,
            )

        # Context from evidence graph
        t0 = time.perf_counter()
        evidence_context = evidence_graph.to_context_string(
            top_k=self.config.rerank_top_k * 2,
        )
        graph_summary = self._build_graph_summary(evidence_graph)
        timings["context_build"] = time.perf_counter() - t0

        # Generation (1 LLM call)
        t0 = time.perf_counter()
        prompt = build_deep_research_prompt(
            question=question,
            evidence_context=evidence_context,
            research_plan=plan,
            graph_summary=graph_summary,
            follow_up_queries=retrieval_result.follow_up_queries,
        )
        answer = self._call_gemini(prompt, DEEP_RESEARCH_SYSTEM)
        timings["generation"] = time.perf_counter() - t0

        # Optional verification
        verified, vnote = True, "Verification skipped."
        if self.config.verify_answer and answer:
            t0 = time.perf_counter()
            verified, vnote = verify_answer(
                answer, evidence_context, question,
                client=self._client, model=self._active_model(),
            )
            if not verified:
                answer = regenerate_strict(
                    question, evidence_context, vnote,
                    client=self._client, model=self._active_model(),
                )
                verified, vnote = True, "Re-generated in strict mode."
            timings["verification"] = time.perf_counter() - t0

        sources = self._extract_sources(final_chunks)
        return PipelineResult(
            answer=answer, sources=sources, rewritten_query=rewritten,
            retrieved_chunks=retrieval_result.all_chunks,
            final_chunks=final_chunks,
            verified=verified, verification_note=vnote,
            timings=timings, research_plan=plan,
            evidence_graph_stats=graph_stats,
            retrieval_iterations=retrieval_result.iteration_count,
            follow_up_queries=retrieval_result.follow_up_queries,
            complexity=complexity, tier="deep",
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self, question: str) -> PipelineResult:
        """Execute the adaptive RAG pipeline."""
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        if not question.strip():
            return PipelineResult(
                answer="Please provide a question.",
                sources=[], rewritten_query=question,
                retrieved_chunks=[], final_chunks=[],
                verified=True, verification_note="",
            )

        # ── Stage 0: Intent classification ─────────────────────────────────
        intent = classify_intent(question)
        if not intent.is_legal:
            # Check if this is a follow-up to a recent legal conversation.
            if self._is_legal_followup(question):
                logger.info(
                    "Follow-up detected (%s) — routing to standard tier",
                    intent.intent,
                )
                # Route through standard tier so the query rewriter can
                # contextualise the vague follow-up using history.
                result = self._run_standard(question, timings)
                ans = result.answer.strip()
                if ans and not ans.startswith("Failed after maximum retries"):
                    self._history.append({"role": "user", "text": question})
                    self._history.append({"role": "assistant", "text": result.answer})
                    if len(self._history) > 20:
                        self._history = self._history[-20:]
                else:
                    if not ans or ans.startswith("Failed after maximum retries"):
                        result.answer = ("I couldn't generate an answer right now "
                                         "due to high demand. Please try again "
                                         "in a moment.")
                        result.sources = []  # don't show sources on failure
                    self._history.append({"role": "user", "text": question})
                    if len(self._history) > 20:
                        self._history = self._history[-20:]
                result.timings["total"] = time.perf_counter() - t_total
                return result

            logger.info("Non-legal (%s) — conversational call", intent.intent)
            answer = self._call_gemini_conversational(question, intent.intent)
            self._history.append({"role": "user", "text": question})
            self._history.append({"role": "assistant", "text": answer})
            return PipelineResult(
                answer=answer, sources=[], rewritten_query=question,
                retrieved_chunks=[], final_chunks=[],
                verified=True,
                verification_note="Non-legal — no verification needed.",
                tier="conversational",
            )

        # ── Adaptive tier routing ──────────────────────────────────────────
        tier = _classify_query_tier(question)
        logger.info("Query tier: %s — %s", tier, question[:80])

        if tier == "fast":
            result = self._run_fast(question, timings)
        elif tier == "deep":
            result = self._run_deep(question, timings)
        else:
            result = self._run_standard(question, timings)

        # ── Update conversation history ────────────────────────────────────
        # Skip storing empty / failure answers to avoid poisoning
        # follow-up context.
        ans = result.answer.strip()
        if ans and not ans.startswith("Failed after maximum retries"):
            self._history.append({"role": "user", "text": question})
            self._history.append({"role": "assistant", "text": result.answer})
            if len(self._history) > 20:
                self._history = self._history[-20:]
        else:
            # Provide user-friendly message on total failure
            if not ans or ans.startswith("Failed after maximum retries"):
                result = PipelineResult(
                    answer=("I couldn't generate an answer right now due to "
                            "high demand. Please try again in a moment."),
                    sources=[],  # don't show sources on failure
                    rewritten_query=result.rewritten_query,
                    retrieved_chunks=result.retrieved_chunks,
                    final_chunks=result.final_chunks,
                    verified=result.verified,
                    verification_note=result.verification_note,
                    timings=result.timings,
                    research_plan=result.research_plan,
                    evidence_graph_stats=result.evidence_graph_stats,
                    retrieval_iterations=result.retrieval_iterations,
                    follow_up_queries=result.follow_up_queries,
                    complexity=result.complexity,
                    tier=result.tier,
                )
            self._history.append({"role": "user", "text": question})
            if len(self._history) > 20:
                self._history = self._history[-20:]

        result.timings["total"] = time.perf_counter() - t_total
        return result

    # ── Streaming variant ──────────────────────────────────────────────────

    def run_stream(self, question: str):
        """
        Generator that yields answer tokens.

        Uses the same adaptive tier routing as ``run()``.
        Verification is always skipped in streaming mode.
        """
        if not question.strip():
            yield "Please provide a question."
            return

        # ── Intent classification ──────────────────────────────────────────
        intent = classify_intent(question)
        if not intent.is_legal:
            # Check if this is a follow-up to a recent legal conversation.
            if self._is_legal_followup(question):
                logger.info(
                    "Stream follow-up detected (%s) — routing to standard",
                    intent.intent,
                )
                # Force standard tier so the query rewriter contextualises
                tier = "standard"
            else:
                logger.info("Non-legal (%s) — conversational stream", intent.intent)
                self._last_sources = []
                self._last_rewritten = question
                self._last_research_plan = None
                self._last_graph_stats = None
                self._last_retrieval_iterations = 0
                self._last_complexity = "n/a"
                self._last_follow_up_queries = []
                self._last_tier = "conversational"
                answer = ""
                for token in self._stream_gemini_conversational(question, intent.intent):
                    answer += token
                    yield token
                # Only store conversational answers that are meaningful
                self._history.append({"role": "user", "text": question})
                if answer.strip():
                    self._history.append({"role": "assistant", "text": answer})
                return
        else:
            tier = _classify_query_tier(question)

        # ── Tier routing ───────────────────────────────────────────────────
        logger.info("Stream tier: %s — %s", tier, question[:80])

        # ── Prepare retrieval query ────────────────────────────────────────
        plan = None
        graph_stats = None
        iterations = 0
        complexity = "simple"
        follow_ups: list[str] = []

        if tier == "fast":
            rewritten = question
            chunks, scores = self._hybrid_retrieve(question)
            context = build_context_block(chunks, scores) if chunks else ""
            prompt = build_advanced_prompt(question, context) if chunks else ""
            system = SYSTEM_INSTRUCTION
        elif tier == "deep":
            rewritten = rewrite_query(
                question,
                conversation_history=self._history[-6:] if self._history else None,
                client=self._client, model=self._active_model(),
            )
            plan = generate_research_plan(
                question=rewritten,
                conversation_history=self._history[-6:] if self._history else None,
                client=self._client, model=self._active_model(),
            )
            complexity = plan.get("complexity", "complex")
            needs_multi_hop = plan.get("requires_multi_hop", False)
            max_iter = self.config.max_retrieval_iterations if needs_multi_hop else 1

            loop_config = RetrievalLoopConfig(
                dense_top_k=self.config.dense_top_k,
                bm25_top_k=self.config.bm25_top_k,
                rrf_k=self.config.rrf_k,
                rerank_top_k=self.config.rerank_top_k,
                max_iterations=max_iter,
                max_total_chunks=self.config.max_total_chunks,
                embedding_model=self.config.embedding_model,
                reranker_model=self.config.reranker_model,
                gemini_model=self._active_model(),
                max_retries=self.config.max_retries,
                base_delay=self.config.base_delay,
            )
            retrieval_result = run_retrieval_loop(
                question=rewritten, plan=plan,
                vector_index=self._vector_index, chunks=self._chunks,
                bm25=self._bm25, client=self._client, config=loop_config,
            )
            chunks = retrieval_result.final_chunks
            scores = []
            graph_stats = retrieval_result.evidence_graph.summary_stats()
            iterations = retrieval_result.iteration_count
            follow_ups = retrieval_result.follow_up_queries

            if chunks:
                evidence_context = retrieval_result.evidence_graph.to_context_string(
                    top_k=self.config.rerank_top_k * 2,
                )
                graph_summary = self._build_graph_summary(retrieval_result.evidence_graph)
                prompt = build_deep_research_prompt(
                    question=question,
                    evidence_context=evidence_context,
                    research_plan=plan,
                    graph_summary=graph_summary,
                    follow_up_queries=follow_ups,
                )
            else:
                context = ""
                prompt = ""
            system = DEEP_RESEARCH_SYSTEM
        else:  # standard
            rewritten = rewrite_query(
                question,
                conversation_history=self._history[-6:] if self._history else None,
                client=self._client, model=self._active_model(),
            )
            chunks, scores = self._hybrid_retrieve(rewritten)
            context = build_context_block(chunks, scores) if chunks else ""
            prompt = build_advanced_prompt(question, context, rewritten) if chunks else ""
            system = SYSTEM_INSTRUCTION

        # ── Store metadata ─────────────────────────────────────────────────
        self._last_sources = self._extract_sources(chunks) if chunks else []
        self._last_rewritten = rewritten
        self._last_research_plan = plan
        self._last_graph_stats = graph_stats
        self._last_retrieval_iterations = iterations
        self._last_complexity = complexity
        self._last_follow_up_queries = follow_ups
        self._last_tier = tier

        if not chunks or not prompt:
            yield "No relevant legal documents were found for your question."
            return

        # ── Stream generation ──────────────────────────────────────────────
        full_answer = ""
        attempt = 0
        max_total = self.config.max_retries + ModelManager.total_models() * ModelManager.total_keys()
        for _ in range(max_total):
            model = self._active_model()
            try:
                response = self._client.models.generate_content_stream(
                    model=model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.2,
                        max_output_tokens=3072,
                    ),
                )
                for chunk in response:
                    if chunk.text:
                        full_answer += chunk.text
                        yield chunk.text
                ModelManager.record_success(model)
                break
            except Exception as exc:
                # If tokens were already yielded in this attempt, signal
                # the frontend to discard them before retrying.
                if full_answer:
                    yield {"event": "retry"}
                    full_answer = ""
                if ModelManager.is_quota_error(exc):
                    ModelManager.mark_exhausted(model)
                    continue  # rotate — does NOT count as a retry
                if ModelManager.is_model_incompatible(exc):
                    ModelManager.mark_exhausted(model)
                    logger.warning("Model %s incompatible — rotating", model)
                    continue  # rotate — does NOT count as a retry
                if ModelManager.is_503_error(exc):
                    rotated = ModelManager.record_503(model)
                    if rotated:
                        continue  # auto-rotated — does NOT count as a retry
                # Transient error → counts as a real retry
                attempt += 1
                if self._is_retryable(exc) and attempt < self.config.max_retries:
                    delay = self.config.base_delay * (2 ** (attempt - 1))
                    logger.warning("Stream error — retry in %ds: %s", delay, exc)
                    yield f"\n⏳ Server busy — retrying in {delay}s…\n"
                    time.sleep(delay)
                    continue
                logger.exception("Streaming error")
                yield f"\n\n[Error: {exc}]"
                return

        # Only store in history if we got a meaningful answer.
        # Empty / failure answers poison follow-up context.
        if full_answer.strip() and not full_answer.startswith("Failed after maximum retries"):
            self._history.append({"role": "user", "text": question})
            self._history.append({"role": "assistant", "text": full_answer})
            if len(self._history) > 20:
                self._history = self._history[-20:]
        else:
            # Even on failure, record the user question so the next
            # follow-up can see what was asked (but no assistant entry).
            if not full_answer.strip():
                full_answer = ("I couldn't generate an answer right now due to "
                               "high demand. Please try again in a moment.")
                yield full_answer
            # Clear sources so the UI doesn't display them for failed answers
            self._last_sources = []
            self._history.append({"role": "user", "text": question})
            if len(self._history) > 20:
                self._history = self._history[-20:]

    # ── Metadata accessors ─────────────────────────────────────────────────

    def get_last_sources(self) -> list[str]:
        return getattr(self, "_last_sources", [])

    def get_last_rewritten_query(self) -> str:
        return getattr(self, "_last_rewritten", "")

    def get_last_research_plan(self) -> dict | None:
        return getattr(self, "_last_research_plan", None)

    def get_last_graph_stats(self) -> dict[str, int] | None:
        return getattr(self, "_last_graph_stats", None)

    def get_last_retrieval_iterations(self) -> int:
        return getattr(self, "_last_retrieval_iterations", 0)

    def get_last_complexity(self) -> str:
        return getattr(self, "_last_complexity", "simple")

    def get_last_follow_up_queries(self) -> list[str]:
        return getattr(self, "_last_follow_up_queries", [])

    def get_last_tier(self) -> str:
        return getattr(self, "_last_tier", "standard")

    def clear_history(self) -> None:
        self._history.clear()
        logger.info("Conversation history cleared")

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        return ModelManager.is_retryable(exc)

    def _active_model(self) -> str:
        """Return the best non-exhausted model, switching API key if needed."""
        model = ModelManager.get_model(self.config.gemini_model)
        # get_model() may have switched the API key — sync the client
        self._sync_client()
        return model

    def _sync_client(self) -> None:
        """Recreate the genai Client if ModelManager switched API keys."""
        current_key = ModelManager.get_active_key()
        if current_key and current_key != self._active_key:
            logger.info("API key rotated → recreating Gemini client (key #%d)",
                        ModelManager.active_key_index() + 1)
            self._client = genai.Client(api_key=current_key)
            self._active_key = current_key

    @staticmethod
    def _extract_sources(chunks: list[Chunk]) -> list[str]:
        seen: set[str] = set()
        sources: list[str] = []
        for c in chunks:
            label = f"{c['act']} — {c['section']}"
            if label not in seen:
                sources.append(label)
                seen.add(label)
        return sources

    @staticmethod
    def _build_graph_summary(evidence_graph) -> str:
        stats = evidence_graph.summary_stats()
        lines = [
            f"Evidence graph: {stats['nodes']} nodes, {stats['edges']} edges, "
            f"{stats['chunks']} unique chunks.",
        ]
        edge_types: dict[str, int] = {}
        for edge in evidence_graph.edges:
            edge_types[edge.relation] = edge_types.get(edge.relation, 0) + 1
        if edge_types:
            lines.append("Relationships found:")
            for rel, count in sorted(edge_types.items(), key=lambda x: -x[1]):
                lines.append(f"  • {rel}: {count}")
        return "\n".join(lines)

    def _no_docs_result(
        self, question: str, rewritten: str, tier: str,
        plan: dict | None = None, graph_stats: dict | None = None,
        iterations: int = 0, complexity: str = "simple",
    ) -> PipelineResult:
        return PipelineResult(
            answer="No relevant legal documents were found for your question.",
            sources=[], rewritten_query=rewritten,
            retrieved_chunks=[], final_chunks=[],
            verified=True, verification_note="No documents to verify.",
            research_plan=plan, evidence_graph_stats=graph_stats,
            retrieval_iterations=iterations,
            complexity=complexity, tier=tier,
        )

    # ── Conversational Gemini (non-legal) ──────────────────────────────────

    _CONVERSATIONAL_SYSTEM = (
        "You are an Indian Legal Assistant chatbot. You ONLY answer questions "
        "about Indian law (IPC, Constitution, CrPC, Hindu Marriage Act, Special "
        "Marriage Act, Dowry Prohibition Act, Domestic Violence Act).\n\n"
        "The user's message is NOT a legal question.\n\n"
        "RULES:\n"
        "1. For greetings / small-talk (hi, hello, how are you, etc.): "
        "   respond warmly and briefly mention you can help with Indian "
        "   legal questions.\n"
        "2. For off-topic questions (science, technology, cooking, sports, "
        "   programming, deep learning, maths, history, etc.): DO NOT "
        "   answer the question. Instead, politely tell the user that "
        "   you are specifically designed for Indian law queries and "
        "   cannot help with that topic. Suggest they ask a legal "
        "   question instead.\n"
        "3. Keep responses concise (2-3 sentences).\n"
        "4. NEVER attempt to answer non-legal questions, even partially.\n"
        "5. Never make up legal information in conversational mode."
    )

    def _call_gemini_conversational(self, question: str, intent: str) -> str:
        attempt = 0
        max_total = self.config.max_retries + ModelManager.total_models() * ModelManager.total_keys()
        for _ in range(max_total):
            model = self._active_model()
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=question,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=self._CONVERSATIONAL_SYSTEM,
                        temperature=0.7,
                        max_output_tokens=256,
                    ),
                )
                ModelManager.record_success(model)
                return (response.text or "").strip()
            except Exception as exc:
                if ModelManager.is_quota_error(exc):
                    ModelManager.mark_exhausted(model)
                    continue  # rotate — does NOT count as a retry
                if ModelManager.is_model_incompatible(exc):
                    ModelManager.mark_exhausted(model)
                    logger.warning("Model %s incompatible — rotating", model)
                    continue  # rotate — does NOT count as a retry
                if ModelManager.is_503_error(exc):
                    rotated = ModelManager.record_503(model)
                    if rotated:
                        continue  # auto-rotated — does NOT count as a retry
                attempt += 1
                if self._is_retryable(exc) and attempt < self.config.max_retries:
                    delay = self.config.base_delay * (2 ** (attempt - 1))
                    logger.warning("Conversational error (%d/%d): %s",
                                   attempt, self.config.max_retries, exc)
                    time.sleep(delay)
                    continue
                logger.exception("Conversational Gemini error")
                return ("I'm here to help with Indian legal questions! "
                        "Feel free to ask me anything about Indian law.")
        return ("I'm here to help with Indian legal questions! "
                "Feel free to ask me anything about Indian law.")

    def _stream_gemini_conversational(self, question: str, intent: str):
        attempt = 0
        max_total = self.config.max_retries + ModelManager.total_models() * ModelManager.total_keys()
        for _ in range(max_total):
            model = self._active_model()
            try:
                response = self._client.models.generate_content_stream(
                    model=model,
                    contents=question,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=self._CONVERSATIONAL_SYSTEM,
                        temperature=0.7,
                        max_output_tokens=256,
                    ),
                )
                for chunk in response:
                    if chunk.text:
                        yield chunk.text
                ModelManager.record_success(model)
                return
            except Exception as exc:
                if ModelManager.is_quota_error(exc):
                    ModelManager.mark_exhausted(model)
                    continue  # rotate — does NOT count as a retry
                if ModelManager.is_model_incompatible(exc):
                    ModelManager.mark_exhausted(model)
                    logger.warning("Model %s incompatible — rotating", model)
                    continue  # rotate — does NOT count as a retry
                if ModelManager.is_503_error(exc):
                    rotated = ModelManager.record_503(model)
                    if rotated:
                        continue  # auto-rotated — does NOT count as a retry
                attempt += 1
                if self._is_retryable(exc) and attempt < self.config.max_retries:
                    delay = self.config.base_delay * (2 ** (attempt - 1))
                    logger.warning("Conversational stream error (%d/%d): %s",
                                   attempt, self.config.max_retries, exc)
                    time.sleep(delay)
                    continue
                logger.exception("Conversational stream error")
                yield ("I'm here to help with Indian legal questions! "
                       "Feel free to ask me anything about Indian law.")
                return
        yield ("I'm here to help with Indian legal questions! "
               "Feel free to ask me anything about Indian law.")

    def _call_gemini(self, prompt: str, system: str) -> str:
        """General-purpose Gemini call with retry + model rotation."""
        attempt = 0
        # Cap total loop iterations to prevent infinite loops
        max_total = self.config.max_retries + ModelManager.total_models() * ModelManager.total_keys()
        for _ in range(max_total):
            model = self._active_model()
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=0.2,
                        max_output_tokens=3072,
                    ),
                )
                ModelManager.record_success(model)
                return (response.text or "").strip()
            except Exception as exc:
                if ModelManager.is_quota_error(exc):
                    ModelManager.mark_exhausted(model)
                    continue  # rotate — does NOT count as a retry
                if ModelManager.is_model_incompatible(exc):
                    ModelManager.mark_exhausted(model)
                    logger.warning("Model %s incompatible — rotating", model)
                    continue  # rotate — does NOT count as a retry
                if ModelManager.is_503_error(exc):
                    rotated = ModelManager.record_503(model)
                    if rotated:
                        continue  # auto-rotated — does NOT count as a retry
                # Transient error → counts as a real retry
                attempt += 1
                if self._is_retryable(exc) and attempt < self.config.max_retries:
                    delay = self.config.base_delay * (2 ** (attempt - 1))
                    logger.warning("Gemini error (%d/%d) — waiting %ds: %s",
                                   attempt, self.config.max_retries, delay, exc)
                    time.sleep(delay)
                    continue
                logger.exception("Gemini API error")
                return f"Error communicating with Gemini: {exc}"
        return "Failed after maximum retries."
