"""
prompt_template.py — Build the final prompt sent to Gemini.

The prompt follows a strict grounding pattern: the model must answer
ONLY from the provided legal context and cite act + section numbers.

Provides:
  • ``build_prompt``           — legacy single-stage prompt
  • ``build_advanced_prompt``  — v2 multi-stage pipeline prompt
  • ``build_deep_research_prompt`` — v3 deep-research agentic prompt
"""

from __future__ import annotations

from rag.retriever import RetrievalResult


# ── System instruction (sent as the Gemini system prompt) ──────────────────
SYSTEM_INSTRUCTION: str = """\
You are an expert Indian legal assistant powered by a multi-stage \
Retrieval-Augmented Generation system.

BEHAVIOUR RULES:
1. Answer the user's question ONLY using the legal context provided below.
2. If the answer is NOT present in the context, respond with:
   "I could not find this information in the provided legal documents. \
Please consult a qualified legal professional."
3. Always cite the **Act name** and **Section number** when referencing \
a provision (e.g. "Indian Penal Code, Section 302").
4. Use clear, professional language understandable by a layperson.
5. When multiple sections are relevant, synthesize them into a coherent \
explanation — do NOT just list them verbatim.
6. Do NOT fabricate or hallucinate any legal provision.
7. Structure longer answers with headings, bullet points, or numbered lists.
8. When a question is ambiguous, briefly state your interpretation before \
answering.
9. For punishment-related questions, always mention: offence, applicable \
section, and the prescribed punishment including imprisonment term and fine.
"""

# ── Deep-research system instruction ──────────────────────────────────────

DEEP_RESEARCH_SYSTEM: str = """\
You are a senior Indian legal research assistant powered by a deep-research \
agentic retrieval system.

You have been given carefully curated legal evidence gathered through a \
multi-step research process: research planning → query decomposition → \
iterative hybrid retrieval → cross-encoder reranking → evidence graph \
construction → gap analysis.

YOUR ROLE:
You are a researcher synthesising evidence into a structured, authoritative \
legal analysis — NOT a simple search tool.

REASONING PROCESS — follow these steps:
1. ANALYSE the gathered evidence carefully.
2. EXTRACT the key legal facts (sections, definitions, punishments, \
   conditions, exceptions).
3. CONNECT related pieces of evidence (e.g. one section defines an \
   offence, another prescribes punishment, another lists exceptions).
4. DERIVE conclusions by applying the law to the question.
5. PRODUCE a structured answer with clear headings / bullet points.

GROUNDING RULES:
1. Every factual claim MUST be supported by the evidence provided.
2. Cite **Act name** and **Section number** for every provision referenced.
3. If evidence is insufficient for any aspect, say so explicitly.
4. Do NOT fabricate or hallucinate any legal provision.
5. When evidence contains conflicting provisions, explain both sides.
6. Distinguish between definitions, offences, punishments, exceptions, \
   and procedures.

ANSWER FORMAT:
• Use markdown headings (##) for major sections.
• Use bullet points for lists of elements / conditions.
• Use bold for section numbers and act names.
• End with a brief **Summary** section.
"""


def build_prompt(
    question: str,
    results: list[RetrievalResult],
) -> str:
    """
    Legacy prompt builder for backward compatibility.
    Assembles the user-turn prompt with retrieved context.
    """
    context_parts: list[str] = []
    for i, res in enumerate(results, 1):
        source = res.source_label
        context_parts.append(
            f"[{i}] {source}\n{res.chunk['text']}"
        )
    context_block = "\n\n".join(context_parts)

    prompt = (
        f"LEGAL CONTEXT:\n"
        f"{'─' * 60}\n"
        f"{context_block}\n"
        f"{'─' * 60}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"ANSWER (cite Act and Section):\n"
    )
    return prompt


def build_advanced_prompt(
    question: str,
    context_block: str,
    rewritten_query: str | None = None,
) -> str:
    """
    Advanced prompt for the multi-stage pipeline.
    """
    parts: list[str] = []

    if rewritten_query and rewritten_query != question:
        parts.append(
            f"RETRIEVAL QUERY (auto-generated from user question):\n"
            f"{rewritten_query}\n"
        )

    parts.append(
        f"LEGAL CONTEXT (retrieved and reranked):\n"
        f"{'─' * 60}\n"
        f"{context_block}\n"
        f"{'─' * 60}\n"
    )

    parts.append(f"USER QUESTION:\n{question}\n")

    parts.append(
        "INSTRUCTIONS FOR YOUR ANSWER:\n"
        "• Synthesize the relevant legal provisions into a clear, structured response.\n"
        "• Cite every Act and Section number you rely on.\n"
        "• If the context contains conflicting provisions, explain both.\n"
        "• If the context is insufficient, state what is missing.\n"
        "• Do NOT copy text verbatim — explain in your own words while preserving legal accuracy.\n"
    )

    parts.append("ANSWER:\n")
    return "\n".join(parts)


def build_deep_research_prompt(
    question: str,
    evidence_context: str,
    research_plan: dict | None = None,
    graph_summary: str | None = None,
    follow_up_queries: list[str] | None = None,
) -> str:
    """
    Build the prompt for the deep-research agentic pipeline.

    Includes the research plan, evidence graph context, and instructions
    for multi-step reasoning and synthesis.
    """
    parts: list[str] = []

    # Research plan summary
    if research_plan:
        analysis = research_plan.get("analysis", "")
        concepts = ", ".join(research_plan.get("concepts", []))
        complexity = research_plan.get("complexity", "unknown")
        tasks = research_plan.get("research_tasks", [])

        parts.append("═══ RESEARCH PLAN ═══")
        parts.append(f"Analysis: {analysis}")
        if concepts:
            parts.append(f"Key concepts: {concepts}")
        parts.append(f"Complexity: {complexity}")
        parts.append(f"Research tasks executed: {len(tasks)}")
        for t in tasks:
            parts.append(f"  • Task {t.get('id', '?')}: {t.get('description', '')}")
        if follow_up_queries:
            parts.append(f"Follow-up queries: {', '.join(follow_up_queries)}")
        parts.append("")

    # Evidence context (from evidence graph)
    parts.append(
        f"═══ CURATED LEGAL EVIDENCE ═══\n"
        f"{'─' * 60}\n"
        f"{evidence_context}\n"
        f"{'─' * 60}\n"
    )

    # Graph relationship summary
    if graph_summary:
        parts.append(f"═══ EVIDENCE RELATIONSHIPS ═══\n{graph_summary}\n")

    # The question
    parts.append(f"═══ USER QUESTION ═══\n{question}\n")

    # Reasoning instructions
    parts.append(
        "═══ INSTRUCTIONS ═══\n"
        "Perform the following reasoning steps:\n"
        "1. ANALYSE: Review all evidence pieces and identify the most relevant ones.\n"
        "2. EXTRACT: Pull out key legal facts — definitions, conditions, punishments, exceptions.\n"
        "3. CONNECT: Link related provisions across different sections and acts.\n"
        "4. REASON: Apply the legal provisions to answer the question.\n"
        "5. SYNTHESISE: Produce a well-structured answer with proper citations.\n\n"
        "Your answer must be:\n"
        "• Grounded in the evidence above (no fabrication)\n"
        "• Clearly structured with headings and bullet points\n"
        "• Citing Act name and Section number for every provision\n"
        "• Acknowledging gaps if evidence is insufficient\n"
    )

    parts.append("═══ ANSWER ═══\n")
    return "\n".join(parts)
