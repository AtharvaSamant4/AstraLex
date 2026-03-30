"""
pipeline.py — Multi-stage RAG pipeline orchestrator.

Wires together every stage of the advanced retrieval-augmented generation
pipeline into a single coherent flow:

    User Question
        ↓
    Query Understanding / Query Rewriting
        ↓
    Hybrid Retrieval (Dense + BM25)
        ↓
    Candidate Expansion (top-N broad retrieval)
        ↓
    Cross-Encoder Reranking
        ↓
    Context Deduplication & Compression
        ↓
    Structured Prompt Construction
        ↓
    LLM Reasoning (Gemini)
        ↓
    Answer Verification
        ↓
    Final Response with Citations
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from google import genai

from rag.bm25_search import BM25Index
from rag.chunker import Chunk
from rag.context_builder import build_context_block, compress_context, deduplicate_chunks
from rag.embedder import embed_query
from rag.intent_classifier import classify_intent
from rag.prompt_template import SYSTEM_INSTRUCTION, build_advanced_prompt
from rag.query_rewriter import rewrite_query
from rag.reranker import rerank
from rag.vectordb import load_index, search_index
from rag.verifier import regenerate_strict, verify_answer

logger = logging.getLogger(__name__)


# ── Pipeline configuration ─────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """All tuneable knobs of the RAG pipeline."""
    # Retrieval
    dense_top_k: int = 20          # broad dense retrieval
    bm25_top_k: int = 20           # broad BM25 retrieval
    rrf_k: int = 60                # reciprocal rank fusion constant

    # Reranking
    rerank_top_k: int = 5          # final chunks after cross-encoder

    # Context
    use_compression: bool = False  # LLM context compression (extra API call)

    # Verification
    verify_answer: bool = True     # post-generation verification

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gemini_model: str = "gemini-3-flash-preview"

    # Retry (covers 429 rate-limit AND 503 server-busy)
    max_retries: int = 5
    base_delay: float = 8.0


# ── Pipeline result ────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Everything the pipeline produces."""
    answer: str
    sources: list[str]
    rewritten_query: str
    retrieved_chunks: list[Chunk]
    final_chunks: list[Chunk]       # after reranking + dedup
    verified: bool
    verification_note: str
    timings: dict[str, float] = field(default_factory=dict)


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────

def _rrf_fuse(
    dense_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """
    Combine two ranked lists via Reciprocal Rank Fusion (RRF).

    RRF score = Σ 1 / (k + rank)  across lists.
    """
    rrf_scores: dict[int, float] = {}

    for rank, (idx, _score) in enumerate(dense_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

    for rank, (idx, _score) in enumerate(bm25_results):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)

    # Sort by fused score descending
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused


# ── Main pipeline class ───────────────────────────────────────────────────

class RAGPipeline:
    """
    Multi-stage RAG pipeline.

    Initialise once (loads index + models), then call ``.run(question)``
    for each user query.
    """

    def __init__(
        self,
        index_dir: str | Path = "index",
        config: PipelineConfig | None = None,
    ) -> None:
        self.config = config or PipelineConfig(
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
            rerank_top_k=int(os.getenv("TOP_K", "5")),
        )
        self.index_dir = Path(index_dir)

        # ── Gemini client ──────────────────────────────────────────────────
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise EnvironmentError("GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=api_key)

        # ── Load FAISS index ───────────────────────────────────────────────
        self._faiss_index, self._chunks = load_index(self.index_dir)

        # ── Load BM25 index ────────────────────────────────────────────────
        self._bm25 = BM25Index.load(self._chunks, self.index_dir)

        # ── Conversation memory (last N turns) ─────────────────────────────
        self._history: list[dict[str, str]] = []

        logger.info(
            "RAG pipeline ready — %d chunks, models: embed=%s, rerank=%s, llm=%s",
            len(self._chunks),
            self.config.embedding_model,
            self.config.reranker_model,
            self.config.gemini_model,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self, question: str) -> PipelineResult:
        """Execute the full multi-stage RAG pipeline."""
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        if not question.strip():
            return PipelineResult(
                answer="Please provide a question.",
                sources=[], rewritten_query=question,
                retrieved_chunks=[], final_chunks=[],
                verified=True, verification_note="",
            )

        # ── Stage 0: Intent classification (skip RAG for chitchat) ─────────
        intent = classify_intent(question)
        if not intent.is_legal:
            logger.info("Non-legal query (%s) — lightweight Gemini call", intent.intent)
            answer = self._call_gemini_conversational(question, intent.intent)
            self._history.append({"role": "user", "text": question})
            self._history.append({"role": "assistant", "text": answer})
            return PipelineResult(
                answer=answer,
                sources=[], rewritten_query=question,
                retrieved_chunks=[], final_chunks=[],
                verified=True, verification_note="Non-legal — no verification needed.",
            )

        # ── Stage 1: Query rewriting ───────────────────────────────────────
        t0 = time.perf_counter()
        rewritten = rewrite_query(
            question,
            conversation_history=self._history[-6:] if self._history else None,
            client=self._client,
            model=self.config.gemini_model,
        )
        timings["query_rewrite"] = time.perf_counter() - t0

        # ── Stage 2: Hybrid retrieval ──────────────────────────────────────
        t0 = time.perf_counter()

        # Dense search
        query_vec = embed_query(rewritten, model_name=self.config.embedding_model)
        dense_scores, dense_indices = search_index(
            self._faiss_index, query_vec, top_k=self.config.dense_top_k,
        )
        dense_results = [
            (int(idx), float(sc))
            for sc, idx in zip(dense_scores[0], dense_indices[0])
            if idx != -1
        ]

        # BM25 search
        bm25_results = self._bm25.search(rewritten, top_k=self.config.bm25_top_k)

        # RRF fusion
        fused = _rrf_fuse(dense_results, bm25_results, k=self.config.rrf_k)
        candidate_indices = [idx for idx, _sc in fused[:40]]  # up to 40 candidates
        candidate_chunks = [self._chunks[i] for i in candidate_indices]

        timings["retrieval"] = time.perf_counter() - t0
        logger.info("Hybrid retrieval: %d dense + %d BM25 → %d fused candidates",
                     len(dense_results), len(bm25_results), len(candidate_chunks))

        if not candidate_chunks:
            return PipelineResult(
                answer="No relevant legal documents were found.",
                sources=[], rewritten_query=rewritten,
                retrieved_chunks=[], final_chunks=[],
                verified=True, verification_note="No documents to verify.",
            )

        # ── Stage 3: Cross-encoder reranking ───────────────────────────────
        t0 = time.perf_counter()
        reranked = rerank(
            query=rewritten,
            chunks=candidate_chunks,
            model_name=self.config.reranker_model,
            top_k=self.config.rerank_top_k,
        )
        final_chunks = [chunk for chunk, _sc in reranked]
        final_scores = [sc for _chunk, sc in reranked]
        timings["reranking"] = time.perf_counter() - t0

        # ── Stage 4: Context deduplication + build ─────────────────────────
        t0 = time.perf_counter()
        final_chunks = deduplicate_chunks(final_chunks)
        context_block = build_context_block(final_chunks, final_scores[:len(final_chunks)])

        if self.config.use_compression:
            context_block = compress_context(
                context_block, question,
                client=self._client, model=self.config.gemini_model,
            )
        timings["context_build"] = time.perf_counter() - t0

        # ── Stage 5: Prompt construction + LLM generation ──────────────────
        t0 = time.perf_counter()
        prompt = build_advanced_prompt(question, context_block, rewritten)
        answer = self._call_gemini(prompt)
        timings["generation"] = time.perf_counter() - t0

        # ── Stage 6: Answer verification ───────────────────────────────────
        verified = True
        verification_note = "Verification skipped."
        if self.config.verify_answer and answer:
            t0 = time.perf_counter()
            verified, verification_note = verify_answer(
                answer, context_block, question,
                client=self._client, model=self.config.gemini_model,
            )
            if not verified:
                logger.info("Verification failed — regenerating with strict mode")
                answer = regenerate_strict(
                    question, context_block, verification_note,
                    client=self._client, model=self.config.gemini_model,
                )
                verified = True
                verification_note = "Re-generated in strict mode after verification failure."
            timings["verification"] = time.perf_counter() - t0

        # ── Build sources ──────────────────────────────────────────────────
        seen: set[str] = set()
        sources: list[str] = []
        for c in final_chunks:
            label = f"{c['act']} — {c['section']}"
            if label not in seen:
                sources.append(label)
                seen.add(label)

        # ── Update conversation history ────────────────────────────────────
        self._history.append({"role": "user", "text": question})
        self._history.append({"role": "assistant", "text": answer})
        # Keep history bounded
        if len(self._history) > 20:
            self._history = self._history[-20:]

        timings["total"] = time.perf_counter() - t_total

        return PipelineResult(
            answer=answer,
            sources=sources,
            rewritten_query=rewritten,
            retrieved_chunks=candidate_chunks,
            final_chunks=final_chunks,
            verified=verified,
            verification_note=verification_note,
            timings=timings,
        )

    # ── Streaming variant ──────────────────────────────────────────────────

    def run_stream(self, question: str):
        """
        Generator that yields answer tokens as they arrive.

        Runs rewrite → retrieval → rerank → context build synchronously,
        then streams the Gemini generation.  Verification is skipped in
        streaming mode to avoid buffering.

        After iteration, call ``get_last_result()`` for metadata.
        """
        if not question.strip():
            yield "Please provide a question."
            return

        # ── Stage 0: Intent classification (skip RAG for chitchat) ─────────
        intent = classify_intent(question)
        if not intent.is_legal:
            logger.info("Non-legal query (%s) — lightweight Gemini call", intent.intent)
            self._last_sources = []
            self._last_rewritten = question
            answer = ""
            for token in self._stream_gemini_conversational(question, intent.intent):
                answer += token
                yield token
            self._history.append({"role": "user", "text": question})
            self._history.append({"role": "assistant", "text": answer})
            return

        # Stages 1-4 (synchronous)
        rewritten = rewrite_query(
            question,
            conversation_history=self._history[-6:] if self._history else None,
            client=self._client,
            model=self.config.gemini_model,
        )

        query_vec = embed_query(rewritten, model_name=self.config.embedding_model)
        dense_scores, dense_indices = search_index(
            self._faiss_index, query_vec, top_k=self.config.dense_top_k,
        )
        dense_results = [
            (int(idx), float(sc))
            for sc, idx in zip(dense_scores[0], dense_indices[0])
            if idx != -1
        ]
        bm25_results = self._bm25.search(rewritten, top_k=self.config.bm25_top_k)
        fused = _rrf_fuse(dense_results, bm25_results, k=self.config.rrf_k)
        candidate_chunks = [self._chunks[idx] for idx, _ in fused[:40]]

        if not candidate_chunks:
            yield "No relevant legal documents were found."
            return

        reranked = rerank(
            query=rewritten, chunks=candidate_chunks,
            model_name=self.config.reranker_model,
            top_k=self.config.rerank_top_k,
        )
        final_chunks = deduplicate_chunks([c for c, _ in reranked])
        context_block = build_context_block(final_chunks)

        prompt = build_advanced_prompt(question, context_block, rewritten)

        # Store metadata
        seen: set[str] = set()
        self._last_sources = []
        for c in final_chunks:
            label = f"{c['act']} — {c['section']}"
            if label not in seen:
                self._last_sources.append(label)
                seen.add(label)
        self._last_rewritten = rewritten

        # Stage 5: Stream generation
        full_answer = ""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self._client.models.generate_content_stream(
                    model=self.config.gemini_model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.2,
                        max_output_tokens=2048,
                    ),
                )
                for chunk in response:
                    if chunk.text:
                        full_answer += chunk.text
                        yield chunk.text
                break
            except Exception as exc:
                if self._is_retryable(exc) and attempt < self.config.max_retries:
                    delay = self.config.base_delay * (2 ** (attempt - 1))
                    logger.warning("Gemini error — retry in %ds: %s", delay, exc)
                    yield f"\n⏳ Server busy — retrying in {delay}s…\n"
                    time.sleep(delay)
                    full_answer = ""  # reset buffer for fresh retry
                    continue
                logger.exception("Gemini streaming error")
                yield f"\n\n[Error: {exc}]"
                return

        # Update history
        self._history.append({"role": "user", "text": question})
        self._history.append({"role": "assistant", "text": full_answer})
        if len(self._history) > 20:
            self._history = self._history[-20:]

    def get_last_sources(self) -> list[str]:
        return getattr(self, "_last_sources", [])

    def get_last_rewritten_query(self) -> str:
        return getattr(self, "_last_rewritten", "")

    def clear_history(self) -> None:
        """Reset conversation memory."""
        self._history.clear()
        logger.info("Conversation history cleared")

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Check if an exception is a retryable Gemini error."""
        msg = str(exc)
        return any(tok in msg for tok in (
            "429", "RESOURCE_EXHAUSTED",
            "503", "UNAVAILABLE",
            "500", "INTERNAL",
            "overloaded", "high demand",
        ))

    # ── Conversational Gemini (for non-legal queries) ──────────────────

    _CONVERSATIONAL_SYSTEM = (
        "You are an Indian Legal Assistant chatbot. You specialise in Indian law "
        "(IPC, Constitution, CrPC, Hindu Marriage Act, Special Marriage Act, "
        "Dowry Prohibition Act, Domestic Violence Act).\n\n"
        "The user's message is NOT a legal question. Respond naturally and "
        "conversationally — be friendly, warm, and helpful, just like a "
        "normal AI assistant would.\n\n"
        "RULES:\n"
        "1. For greetings / small-talk: respond warmly and briefly mention "
        "   you can help with Indian legal questions.\n"
        "2. For off-topic questions (science, cooking, sports, etc.): "
        "   politely let them know your expertise is in Indian law, but "
        "   still be friendly — don't just refuse.\n"
        "3. Keep responses concise (2-4 sentences).\n"
        "4. Never make up legal information in conversational mode."
    )

    def _call_gemini_conversational(self, question: str, intent: str) -> str:
        """Lightweight Gemini call for non-legal queries (no RAG context)."""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=question,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=self._CONVERSATIONAL_SYSTEM,
                        temperature=0.7,
                        max_output_tokens=256,
                    ),
                )
                return (response.text or "").strip()
            except Exception as exc:
                if self._is_retryable(exc) and attempt < self.config.max_retries:
                    delay = self.config.base_delay * (2 ** (attempt - 1))
                    logger.warning("Conversational Gemini error (attempt %d/%d): %s",
                                   attempt, self.config.max_retries, exc)
                    time.sleep(delay)
                    continue
                logger.exception("Conversational Gemini error")
                return "I'm here to help with Indian legal questions! Feel free to ask me anything about Indian law."
        return "I'm here to help with Indian legal questions! Feel free to ask me anything about Indian law."

    def _stream_gemini_conversational(self, question: str, intent: str):
        """Streaming variant of conversational Gemini."""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self._client.models.generate_content_stream(
                    model=self.config.gemini_model,
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
                return
            except Exception as exc:
                if self._is_retryable(exc) and attempt < self.config.max_retries:
                    delay = self.config.base_delay * (2 ** (attempt - 1))
                    logger.warning("Conversational stream error (attempt %d/%d): %s",
                                   attempt, self.config.max_retries, exc)
                    time.sleep(delay)
                    continue
                logger.exception("Conversational stream error")
                yield "I'm here to help with Indian legal questions! Feel free to ask me anything about Indian law."
                return
        yield "I'm here to help with Indian legal questions! Feel free to ask me anything about Indian law."

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini with retry logic."""
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.config.gemini_model,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.2,
                        max_output_tokens=2048,
                    ),
                )
                return (response.text or "").strip()
            except Exception as exc:
                if self._is_retryable(exc) and attempt < self.config.max_retries:
                    delay = self.config.base_delay * (2 ** (attempt - 1))
                    logger.warning("Gemini error (attempt %d/%d) — waiting %ds: %s",
                                   attempt, self.config.max_retries, delay, exc)
                    time.sleep(delay)
                    continue
                logger.exception("Gemini API error")
                return f"Error communicating with Gemini: {exc}"
        return "Failed after maximum retries."
