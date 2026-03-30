"""
chatbot.py — High-level chatbot interface backed by the deep-research
agentic RAG pipeline.

This is the public API that the CLI and FastAPI server use.  It delegates
all heavy lifting to ``rag.pipeline.RAGPipeline``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env once at module import
load_dotenv()


@dataclass
class ChatResponse:
    """Structured response returned by the chatbot."""
    answer: str
    sources: list[str]
    rewritten_query: str
    verified: bool
    verification_note: str
    timings: dict[str, float]

    # Deep-research metadata
    research_plan: dict | None = None
    evidence_graph_stats: dict[str, int] | None = None
    complexity: str = "simple"
    retrieval_iterations: int = 1
    follow_up_queries: list[str] = field(default_factory=list)


class LegalChatbot:
    """
    End-to-end legal RAG chatbot using the deep-research agentic pipeline.

    Pipeline stages:
      0. Intent classification (skip RAG for non-legal)
      1. Query rewriting (context-aware LLM)
      2. Research planning (LLM generates structured plan)
      3. Iterative hybrid retrieval loop
         (FAISS + BM25 → RRF → reranking → gap analysis)
      4. Evidence graph construction
      5. Deep-research prompt → Gemini reasoning
      6. Self-verification & optional strict regeneration

    Usage::

        bot = LegalChatbot()
        resp = bot.ask("What is the punishment for murder?")
        print(resp.answer)
    """

    def __init__(
        self,
        index_dir: str | Path = "index",
    ) -> None:
        # Lazy import to keep startup fast when just importing the module
        from rag.pipeline import PipelineConfig, RAGPipeline

        config = PipelineConfig(
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            rerank_top_k=int(os.getenv("TOP_K", "5")),
        )
        self._pipeline = RAGPipeline(index_dir=index_dir, config=config)
        logger.info("LegalChatbot initialised with deep-research agentic pipeline")

    # ── Public API ─────────────────────────────────────────────────────────

    def ask(self, question: str) -> ChatResponse:
        """
        Answer a legal question through the deep-research agentic pipeline.
        """
        result = self._pipeline.run(question)
        return ChatResponse(
            answer=result.answer,
            sources=result.sources,
            rewritten_query=result.rewritten_query,
            verified=result.verified,
            verification_note=result.verification_note,
            timings=result.timings,
            research_plan=result.research_plan,
            evidence_graph_stats=result.evidence_graph_stats,
            complexity=result.complexity,
            retrieval_iterations=result.retrieval_iterations,
            follow_up_queries=result.follow_up_queries,
        )

    def ask_stream(self, question: str):
        """
        Streaming variant — yields answer tokens as they arrive from Gemini.

        Runs the full deep-research pipeline synchronously (planning →
        iterative retrieval → evidence graph), then streams generation.
        Verification is skipped in streaming mode.

        After iteration, call ``get_last_sources()`` etc. for metadata.
        """
        yield from self._pipeline.run_stream(question)

    def get_last_sources(self) -> list[str]:
        """Return sources from the most recent ``ask_stream`` call."""
        return self._pipeline.get_last_sources()

    def get_last_rewritten_query(self) -> str:
        """Return the rewritten query from the most recent streaming call."""
        return self._pipeline.get_last_rewritten_query()

    def get_last_research_plan(self) -> dict | None:
        """Return the research plan from the most recent streaming call."""
        return self._pipeline.get_last_research_plan()

    def get_last_graph_stats(self) -> dict[str, int] | None:
        """Return evidence graph stats from the most recent streaming call."""
        return self._pipeline.get_last_graph_stats()

    def get_last_retrieval_iterations(self) -> int:
        """Return retrieval iteration count from the most recent streaming call."""
        return self._pipeline.get_last_retrieval_iterations()

    def get_last_complexity(self) -> str:
        """Return complexity level from the most recent streaming call."""
        return self._pipeline.get_last_complexity()

    def get_last_follow_up_queries(self) -> list[str]:
        """Return follow-up queries from the most recent streaming call."""
        return self._pipeline.get_last_follow_up_queries()

    def get_last_tier(self) -> str:
        """Return tier from the most recent streaming call."""
        return self._pipeline.get_last_tier()

    def clear_history(self) -> None:
        """Reset conversation memory."""
        self._pipeline.clear_history()
