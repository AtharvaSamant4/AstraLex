#!/usr/bin/env python3
"""
test_suite.py — Comprehensive manual evaluation of the Deep-Research
Agentic RAG chatbot.

Runs all test categories, captures answers + metadata, and writes a
detailed report to ``test_results.txt``.

Usage:
    python test_suite.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Keep console quiet — only warnings+
logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

# ── Test cases ─────────────────────────────────────────────────────────────

TEST_CASES: list[dict] = [
    # ── 1. Direct Relevant (Basic Retrieval) ───────────────────────────────
    {"id": "1.1", "cat": "Direct Relevant", "q": "What is the punishment for murder in India?"},
    {"id": "1.2", "cat": "Direct Relevant", "q": "What is the punishment for attempt to murder?"},
    {"id": "1.3", "cat": "Direct Relevant", "q": "What does IPC Section 302 state?"},
    {"id": "1.4", "cat": "Direct Relevant", "q": "What does IPC Section 307 say?"},
    {"id": "1.5", "cat": "Direct Relevant", "q": "What is the punishment for dowry harassment?"},
    {"id": "1.6", "cat": "Direct Relevant", "q": "What is the punishment for domestic violence?"},
    {"id": "1.7", "cat": "Direct Relevant", "q": "What is the legal age of marriage in India?"},
    {"id": "1.8", "cat": "Direct Relevant", "q": "What is the punishment for kidnapping?"},
    {"id": "1.9", "cat": "Direct Relevant", "q": "What does the Constitution say about freedom of speech?"},
    {"id": "1.10", "cat": "Direct Relevant", "q": "What is the punishment for theft?"},

    # ── 2. Section-Based Queries ───────────────────────────────────────────
    {"id": "2.1", "cat": "Section-Based", "q": "Explain IPC Section 300."},
    {"id": "2.2", "cat": "Section-Based", "q": "What does Section 498A of IPC state?"},
    {"id": "2.3", "cat": "Section-Based", "q": "What is written in Article 21 of the Constitution?"},
    {"id": "2.4", "cat": "Section-Based", "q": "What does Article 14 guarantee?"},
    {"id": "2.5", "cat": "Section-Based", "q": "Explain Section 125 of CrPC."},

    # ── 3. Multi-Hop Questions ─────────────────────────────────────────────
    {"id": "3.1", "cat": "Multi-Hop", "q": "What is the difference between murder and culpable homicide?"},
    {"id": "3.2", "cat": "Multi-Hop", "q": "What is the difference between murder and attempt to murder?"},
    {"id": "3.3", "cat": "Multi-Hop", "q": "What is the difference between IPC 299 and 300?"},
    {"id": "3.4", "cat": "Multi-Hop", "q": "How does Indian law define murder and what punishment does it prescribe?"},
    {"id": "3.5", "cat": "Multi-Hop", "q": "Compare punishment for theft and robbery."},

    # ── 4. Complex Reasoning ───────────────────────────────────────────────
    {"id": "4.1", "cat": "Complex Reasoning", "q": "If someone intentionally kills another person, what law applies?"},
    {"id": "4.2", "cat": "Complex Reasoning", "q": "What happens legally if someone attempts murder but fails?"},
    {"id": "4.3", "cat": "Complex Reasoning", "q": "What legal protections exist for women facing domestic violence?"},
    {"id": "4.4", "cat": "Complex Reasoning", "q": "What legal rights protect individuals from unlawful arrest?"},
    {"id": "4.5", "cat": "Complex Reasoning", "q": "If someone steals property worth a large amount, what punishment applies?"},

    # ── 5. Ambiguous Questions ─────────────────────────────────────────────
    {"id": "5.1", "cat": "Ambiguous", "q": "murder punishment"},
    {"id": "5.2", "cat": "Ambiguous", "q": "attempt murder law"},
    {"id": "5.3", "cat": "Ambiguous", "q": "what happens if someone kills someone"},
    {"id": "5.4", "cat": "Ambiguous", "q": "dowry law india"},
    {"id": "5.5", "cat": "Ambiguous", "q": "marriage law age"},

    # ── 6. Vague Questions ─────────────────────────────────────────────────
    {"id": "6.1", "cat": "Vague", "q": "Tell me about murder law."},
    {"id": "6.2", "cat": "Vague", "q": "What happens if someone commits a crime?"},
    {"id": "6.3", "cat": "Vague", "q": "What does Indian law say about violence?"},
    {"id": "6.4", "cat": "Vague", "q": "Explain marriage laws."},
    {"id": "6.5", "cat": "Vague", "q": "What happens if someone harms someone?"},

    # ── 7. Irrelevant (Out-of-Scope) ───────────────────────────────────────
    {"id": "7.1", "cat": "Irrelevant", "q": "What is the capital of France?"},
    {"id": "7.2", "cat": "Irrelevant", "q": "Who won the FIFA World Cup in 2018?"},
    {"id": "7.3", "cat": "Irrelevant", "q": "What is the best programming language?"},
    {"id": "7.4", "cat": "Irrelevant", "q": "How do I cook pasta?"},
    {"id": "7.5", "cat": "Irrelevant", "q": "What is quantum physics?"},

    # ── 8. Adversarial (Hallucination Resistance) ──────────────────────────
    {"id": "8.1", "cat": "Adversarial", "q": "What is IPC Section 9999?"},
    {"id": "8.2", "cat": "Adversarial", "q": "Does Indian law allow legal murder?"},
    {"id": "8.3", "cat": "Adversarial", "q": "What punishment exists for stealing oxygen?"},
    {"id": "8.4", "cat": "Adversarial", "q": "What does Article 420 of the Constitution say?"},
    {"id": "8.5", "cat": "Adversarial", "q": "What is the punishment for insulting aliens?"},

    # ── 9. Misleading ──────────────────────────────────────────────────────
    {"id": "9.1", "cat": "Misleading", "q": "What punishment exists for legal murder under IPC?"},
    {"id": "9.2", "cat": "Misleading", "q": "Which IPC section allows killing someone in anger?"},
    {"id": "9.3", "cat": "Misleading", "q": "Why does Indian law allow dowry?"},
    {"id": "9.4", "cat": "Misleading", "q": "Which law makes theft legal?"},
    {"id": "9.5", "cat": "Misleading", "q": "Which article of the constitution bans freedom of speech?"},

    # ── 12. Completely Dumb Questions ──────────────────────────────────────
    {"id": "12.1", "cat": "Dumb", "q": "If I punch the sun will IPC arrest me?"},
    {"id": "12.2", "cat": "Dumb", "q": "Can I marry a robot under Indian law?"},
    {"id": "12.3", "cat": "Dumb", "q": "What happens if I steal the moon?"},
    {"id": "12.4", "cat": "Dumb", "q": "Is murder allowed on Tuesdays?"},
    {"id": "12.5", "cat": "Dumb", "q": "Can ghosts be punished under IPC?"},

    # ── 14. Hallucination Trap ─────────────────────────────────────────────
    {"id": "14.1", "cat": "Hallucination Trap", "q": "IPC Section 12345"},
    {"id": "14.2", "cat": "Hallucination Trap", "q": "Article 999 of Constitution"},
    {"id": "14.3", "cat": "Hallucination Trap", "q": "Domestic Violence Act Section 5000"},

    # ── 11. Stress Test (Long Messy Queries) ───────────────────────────────
    {"id": "11.1", "cat": "Stress Test", "q": "If a person intentionally tries to kill another person but the victim survives due to medical intervention, which law applies and what punishment can the offender receive?"},
    {"id": "11.2", "cat": "Stress Test", "q": "Under Indian criminal law, how is murder defined and how does it differ from culpable homicide not amounting to murder?"},
    {"id": "11.3", "cat": "Stress Test", "q": "What legal provisions exist for protecting women from dowry harassment and domestic violence?"},

    # ── Ultimate Test ──────────────────────────────────────────────────────
    {"id": "ULTIMATE", "cat": "Ultimate", "q": "Explain the difference between murder, culpable homicide, and attempt to murder, including their legal definitions and punishments under Indian law."},
]

# ── Follow-up conversation test (category 10) — run separately ─────────
FOLLOWUP_TESTS = [
    {"id": "10.1", "q": "What is punishment for murder?"},
    {"id": "10.2", "q": "What about attempt?"},
    {"id": "10.3", "q": "Which section is that?"},
    {"id": "10.4", "q": "Explain it simply."},
]

# ── Latency tests (category 13) ───────────────────────────────────────────
LATENCY_TESTS = [
    {"id": "13.1", "q": "What is IPC 302?"},
    {"id": "13.2", "q": "What is IPC 307?"},
    {"id": "13.3", "q": "What is IPC 299?"},
    {"id": "13.4", "q": "What is IPC 300?"},
    {"id": "13.5", "q": "What is IPC 498A?"},
]


def _grade(test_case: dict, answer: str, sources: list[str]) -> str:
    """Auto-grade based on category expectations."""
    cat = test_case["cat"]
    ans_lower = answer.lower()

    if cat == "Irrelevant":
        # Should NOT answer the question — should redirect to legal topics
        keywords = ["capital", "france", "fifa", "programming", "pasta", "quantum",
                     "paris", "java", "python", "c++", "boil", "noodle"]
        answered = any(k in ans_lower for k in keywords)
        if answered:
            return "FAIL — answered an irrelevant question"
        if "legal" in ans_lower or "law" in ans_lower or "indian" in ans_lower:
            return "PASS — redirected to legal domain"
        return "WARN — unclear response"

    if cat == "Adversarial" or cat == "Hallucination Trap":
        # Should say "no such section/provision" or "not found"
        hallucination_markers = [
            "section 9999", "article 420 of the constitution", "stealing oxygen",
            "insulting aliens", "section 12345", "article 999", "section 5000",
        ]
        not_found_indicators = [
            "could not find", "no such", "does not exist", "not found",
            "no provision", "not present", "insufficient", "no information",
            "not available", "no specific"
        ]
        has_not_found = any(i in ans_lower for i in not_found_indicators)
        if has_not_found:
            return "PASS — correctly indicated no such provision"
        return "WARN — may have hallucinated (review needed)"

    if cat == "Misleading":
        # Should correct the premise
        correction_indicators = [
            "does not allow", "no such", "prohibits", "illegal", "not permitted",
            "there is no", "indian law does not", "misconception", "incorrect",
            "contrary", "rather", "actually", "in fact", "however",
        ]
        has_correction = any(i in ans_lower for i in correction_indicators)
        if has_correction:
            return "PASS — corrected the misleading premise"
        return "WARN — may not have corrected premise (review needed)"

    if cat == "Dumb":
        not_applicable = [
            "does not", "no provision", "not applicable", "absurd",
            "no such", "not possible", "cannot", "does not correspond",
            "no legal provision", "not recognized", "hypothetical",
        ]
        has_appropriate = any(i in ans_lower for i in not_applicable)
        if has_appropriate:
            return "PASS — handled absurd question appropriately"
        return "WARN — review needed"

    if cat in ("Direct Relevant", "Section-Based", "Multi-Hop",
               "Complex Reasoning", "Stress Test", "Ultimate"):
        # Should have sources and mention sections
        has_sources = len(sources) > 0
        mentions_section = "section" in ans_lower or "article" in ans_lower
        if has_sources and mentions_section:
            return "PASS — has sources and cites sections"
        if has_sources:
            return "WARN — has sources but may not cite sections"
        return "FAIL — no sources retrieved"

    if cat in ("Ambiguous", "Vague"):
        has_sources = len(sources) > 0
        if has_sources:
            return "PASS — expanded query and retrieved relevant info"
        return "WARN — no sources (may not have understood query)"

    return "SKIP — no auto-grade rule"


def run_tests():
    """Run all test categories and write results to file."""
    from chatbot.chatbot import LegalChatbot

    report_path = Path("test_results.txt")
    bot = LegalChatbot(index_dir="index")

    results: list[dict] = []
    total_pass = 0
    total_fail = 0
    total_warn = 0

    def _run_one(tc: dict, bot_instance, clear_first=False):
        nonlocal total_pass, total_fail, total_warn
        if clear_first:
            bot_instance.clear_history()

        qid = tc["id"]
        cat = tc.get("cat", "Follow-Up")
        q = tc["q"]

        print(f"\n{'='*70}")
        print(f"[{qid}] ({cat}) {q}")
        print(f"{'='*70}")

        t0 = time.perf_counter()
        try:
            resp = bot_instance.ask(q)
            elapsed = time.perf_counter() - t0
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  ERROR: {exc}")
            result = {
                "id": qid, "cat": cat, "question": q,
                "answer": f"ERROR: {exc}", "sources": [],
                "rewritten_query": "", "complexity": "",
                "iterations": 0, "graph_stats": {},
                "timings": {}, "elapsed": elapsed,
                "grade": "FAIL — exception",
            }
            results.append(result)
            total_fail += 1
            return

        grade = _grade(tc, resp.answer, resp.sources)
        if grade.startswith("PASS"):
            total_pass += 1
            grade_icon = "✅"
        elif grade.startswith("FAIL"):
            total_fail += 1
            grade_icon = "❌"
        else:
            total_warn += 1
            grade_icon = "⚠️"

        # Print summary
        answer_preview = resp.answer[:200].replace("\n", " ")
        print(f"  {grade_icon} {grade}")
        print(f"  Answer: {answer_preview}…")
        print(f"  Sources: {resp.sources}")
        print(f"  Complexity: {resp.complexity} | "
              f"Iterations: {resp.retrieval_iterations} | "
              f"Time: {elapsed:.1f}s")
        if resp.timings:
            for k, v in resp.timings.items():
                if k != "total":
                    print(f"    {k}: {v:.2f}s")

        result = {
            "id": qid, "cat": cat, "question": q,
            "answer": resp.answer, "sources": resp.sources,
            "rewritten_query": resp.rewritten_query,
            "complexity": resp.complexity,
            "iterations": resp.retrieval_iterations,
            "graph_stats": resp.evidence_graph_stats or {},
            "timings": resp.timings,
            "elapsed": elapsed,
            "grade": grade,
        }
        results.append(result)

        # Rate-limit pause (gemini-2.5-flash: 15 RPM)
        # Each question uses ~5 API calls → wait ~20s between questions
        time.sleep(5)

    # ── Run main test cases ────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  MAIN TEST SUITE")
    print("█" * 70)

    for tc in TEST_CASES:
        _run_one(tc, bot, clear_first=True)

    # ── Run follow-up conversation test ────────────────────────────────────
    print("\n" + "█" * 70)
    print("  FOLLOW-UP CONVERSATION TEST (Category 10)")
    print("█" * 70)
    bot.clear_history()
    for tc in FOLLOWUP_TESTS:
        tc["cat"] = "Follow-Up"
        _run_one(tc, bot, clear_first=False)  # Don't clear — test memory

    # ── Run latency tests ─────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  LATENCY TESTS (Category 13)")
    print("█" * 70)
    for tc in LATENCY_TESTS:
        tc["cat"] = "Latency"
        _run_one(tc, bot, clear_first=True)

    # ── Write detailed report ──────────────────────────────────────────────
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  DEEP-RESEARCH AGENTIC RAG CHATBOT — TEST REPORT\n")
        f.write(f"  Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"SUMMARY: {total_pass} PASS | {total_fail} FAIL | {total_warn} WARN\n")
        f.write(f"Total tests: {len(results)}\n\n")

        # Timing stats
        legal_times = [r["elapsed"] for r in results
                       if r["cat"] not in ("Irrelevant",) and r["elapsed"] > 1]
        if legal_times:
            f.write(f"Avg legal question time: {sum(legal_times)/len(legal_times):.1f}s\n")
            f.write(f"Min: {min(legal_times):.1f}s | Max: {max(legal_times):.1f}s\n\n")

        current_cat = ""
        for r in results:
            if r["cat"] != current_cat:
                current_cat = r["cat"]
                f.write("\n" + "─" * 80 + "\n")
                f.write(f"  CATEGORY: {current_cat}\n")
                f.write("─" * 80 + "\n")

            f.write(f"\n[{r['id']}] Q: {r['question']}\n")
            f.write(f"GRADE: {r['grade']}\n")
            f.write(f"Rewritten: {r.get('rewritten_query', '')}\n")
            f.write(f"Complexity: {r.get('complexity', '')} | "
                    f"Iterations: {r.get('iterations', 0)} | "
                    f"Time: {r['elapsed']:.1f}s\n")
            f.write(f"Sources: {r['sources']}\n")
            f.write(f"Graph: {r.get('graph_stats', {})}\n")
            f.write(f"ANSWER:\n{r['answer']}\n")

    # Also save as JSON for programmatic analysis
    json_path = Path("test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print("\n\n" + "█" * 70)
    print(f"  FINAL SCORE: {total_pass} PASS | {total_fail} FAIL | {total_warn} WARN")
    print(f"  Total: {len(results)} tests")
    print(f"  Report: {report_path.resolve()}")
    print(f"  JSON:   {json_path.resolve()}")
    print("█" * 70)


if __name__ == "__main__":
    run_tests()
