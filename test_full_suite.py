"""
test_full_suite.py — Comprehensive test runner for the Legal RAG Chatbot.

Runs 60+ questions across 14 categories, captures answers, timing,
tier routing, and model rotation data. Saves results to test_results.json
and prints a human-readable report.
"""

import json
import os
import sys
import time
import threading
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Manual .env loading
    if os.path.exists(".env"):
        for line in open(".env"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# Ensure GEMINI_API_KEY is set
if not os.getenv("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY not set in .env")
    sys.exit(1)

from rag.pipeline import RAGPipeline
from rag.model_manager import ModelManager

# Maximum time per question before we skip it
QUESTION_TIMEOUT_SECONDS = 120


class _TimeoutResult:
    """Container for cross-thread result passing."""
    def __init__(self):
        self.result = None
        self.error = None


def _run_with_timeout(pipeline, question, timeout):
    """Run pipeline.run(question) with a timeout. Returns (result, elapsed)."""
    container = _TimeoutResult()

    def _worker():
        try:
            container.result = pipeline.run(question)
        except Exception as e:
            container.error = e

    t0 = time.time()
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    elapsed = time.time() - t0

    if thread.is_alive():
        # Thread is still running — timed out
        raise TimeoutError(f"Question timed out after {timeout}s")

    if container.error:
        raise container.error

    return container.result, elapsed

# ── Test questions organized by category ───────────────────────────────────

TESTS = {
    "1. Direct Relevant (Basic Retrieval)": [
        "What is the punishment for murder in India?",
        "What is the punishment for attempt to murder?",
        "What does IPC Section 302 state?",
        "What does IPC Section 307 say?",
        "What is the punishment for dowry harassment?",
        "What is the punishment for domestic violence?",
        "What is the legal age of marriage in India?",
        "What is the punishment for kidnapping?",
        "What does the Constitution say about freedom of speech?",
        "What is the punishment for theft?",
    ],
    "2. Section-Based Queries": [
        "Explain IPC Section 300.",
        "What does Section 498A of IPC state?",
        "What is written in Article 21 of the Constitution?",
        "What does Article 14 guarantee?",
        "Explain Section 125 of CrPC.",
    ],
    "3. Multi-Hop Questions": [
        "What is the difference between murder and culpable homicide?",
        "What is the difference between murder and attempt to murder?",
        "What is the difference between IPC 299 and 300?",
        "How does Indian law define murder and what punishment does it prescribe?",
        "Compare punishment for theft and robbery.",
    ],
    "4. Complex Reasoning": [
        "If someone intentionally kills another person, what law applies?",
        "What happens legally if someone attempts murder but fails?",
        "What legal protections exist for women facing domestic violence?",
        "What legal rights protect individuals from unlawful arrest?",
        "If someone steals property worth a large amount, what punishment applies?",
    ],
    "5. Ambiguous Queries": [
        "murder punishment",
        "attempt murder law",
        "what happens if someone kills someone",
        "dowry law india",
        "marriage law age",
    ],
    "6. Vague Questions": [
        "Tell me about murder law.",
        "What happens if someone commits a crime?",
        "What does Indian law say about violence?",
        "Explain marriage laws.",
        "What happens if someone harms someone?",
    ],
    "7. Irrelevant (Out-of-Scope)": [
        "What is the capital of France?",
        "Who won the FIFA World Cup in 2018?",
        "What is the best programming language?",
        "How do I cook pasta?",
        "What is quantum physics?",
    ],
    "8. Adversarial (Hallucination Resistance)": [
        "What is IPC Section 9999?",
        "Does Indian law allow legal murder?",
        "What punishment exists for stealing oxygen?",
        "What does Article 420 of the Constitution say?",
        "What is the punishment for insulting aliens?",
    ],
    "9. Misleading Questions": [
        "What punishment exists for legal murder under IPC?",
        "Which IPC section allows killing someone in anger?",
        "Why does Indian law allow dowry?",
        "Which law makes theft legal?",
        "Which article of the constitution bans freedom of speech?",
    ],
    "10. Follow-Up Conversation": [
        "What is punishment for murder?",
        "What about attempt?",
        "Which section is that?",
        "Explain it simply.",
    ],
    "11. Stress Test (Long Queries)": [
        "If a person intentionally tries to kill another person but the victim survives due to medical intervention, which law applies and what punishment can the offender receive?",
        "Under Indian criminal law, how is murder defined and how does it differ from culpable homicide not amounting to murder?",
        "What legal provisions exist for protecting women from dowry harassment and domestic violence?",
    ],
    "12. Dumb Questions": [
        "If I punch the sun will IPC arrest me?",
        "Can I marry a robot under Indian law?",
        "What happens if I steal the moon?",
        "Is murder allowed on Tuesdays?",
        "Can ghosts be punished under IPC?",
    ],
    "13. Latency Stress": [
        "What is IPC 302?",
        "What is IPC 307?",
        "What is IPC 299?",
        "What is IPC 300?",
        "What is IPC 498A?",
    ],
    "14. Hallucination Trap": [
        "IPC Section 12345",
        "Article 999 of Constitution",
        "Domestic Violence Act Section 5000",
    ],
    "15. Ultimate Test": [
        "Explain the difference between murder, culpable homicide, and attempt to murder, including their legal definitions and punishments under Indian law.",
    ],
}


def run_tests():
    print("=" * 70)
    print("  LEGAL RAG CHATBOT — COMPREHENSIVE TEST SUITE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Reset model rotation state for clean test run
    ModelManager.reset()
    print(f"  Models available: {ModelManager.total_models()} (rotation reset)")
    print("=" * 70)
    print()

    pipeline = RAGPipeline()
    all_results = []
    category_summaries = []
    total_questions = sum(len(qs) for qs in TESTS.values())
    question_num = 0

    for category, questions in TESTS.items():
        print(f"\n{'━' * 70}")
        print(f"  CATEGORY: {category}")
        print(f"{'━' * 70}")

        cat_results = []

        for q in questions:
            question_num += 1
            print(f"\n┌─ Q{question_num}/{total_questions}: {q}")
            print(f"│  Models exhausted: {ModelManager.exhausted_count()}/{ModelManager.total_models()}")

            t0 = time.time()
            try:
                result, elapsed = _run_with_timeout(pipeline, q, QUESTION_TIMEOUT_SECONDS)

                answer = result.answer
                tier = result.tier
                sources = result.sources[:3]
                timings = result.timings

                # Truncate answer for display
                display_answer = answer[:400] + "..." if len(answer) > 400 else answer

                print(f"│  Tier: {tier} | Time: {elapsed:.1f}s | Sources: {len(result.sources)}")
                print(f"│  Timings: ", end="")
                for k, v in sorted(timings.items()):
                    print(f"{k}={v:.1f}s ", end="")
                print()
                print(f"│")
                for line in display_answer.split("\n")[:8]:
                    print(f"│  {line}")
                if len(display_answer.split("\n")) > 8:
                    print(f"│  ... (truncated)")
                print(f"└─ ✓")

                entry = {
                    "category": category,
                    "question": q,
                    "answer": answer,
                    "tier": tier,
                    "sources": sources,
                    "time_seconds": round(elapsed, 2),
                    "timings": {k: round(v, 2) for k, v in timings.items()},
                    "models_exhausted": ModelManager.exhausted_count(),
                    "answer_length": len(answer),
                    "status": "OK",
                }

            except Exception as exc:
                elapsed = time.time() - t0
                print(f"│  ✗ ERROR after {elapsed:.1f}s: {exc}")
                print(f"└─")
                entry = {
                    "category": category,
                    "question": q,
                    "answer": f"ERROR: {exc}",
                    "tier": "error",
                    "sources": [],
                    "time_seconds": round(elapsed, 2),
                    "timings": {},
                    "models_exhausted": ModelManager.exhausted_count(),
                    "answer_length": 0,
                    "status": "ERROR",
                }

            cat_results.append(entry)
            all_results.append(entry)

            # Small delay between questions to be polite to the API
            time.sleep(0.5)

        # Category summary
        ok = sum(1 for r in cat_results if r["status"] == "OK")
        avg_time = sum(r["time_seconds"] for r in cat_results) / len(cat_results) if cat_results else 0
        tiers = {}
        for r in cat_results:
            tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1

        summary = {
            "category": category,
            "total": len(cat_results),
            "ok": ok,
            "errors": len(cat_results) - ok,
            "avg_time": round(avg_time, 2),
            "tiers": tiers,
        }
        category_summaries.append(summary)
        print(f"\n  📊 {category}: {ok}/{len(cat_results)} OK, avg {avg_time:.1f}s, tiers: {tiers}")

    # ── Final report ───────────────────────────────────────────────────────
    print(f"\n\n{'=' * 70}")
    print("  FINAL REPORT")
    print(f"{'=' * 70}")

    total_ok = sum(s["ok"] for s in category_summaries)
    total_err = sum(s["errors"] for s in category_summaries)
    total_time = sum(r["time_seconds"] for r in all_results)
    avg_time = total_time / len(all_results) if all_results else 0

    print(f"\n  Total questions: {len(all_results)}")
    print(f"  Successful: {total_ok}")
    print(f"  Errors: {total_err}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  Avg time/question: {avg_time:.1f}s")
    print(f"  Models exhausted: {ModelManager.exhausted_count()}/{ModelManager.total_models()}")

    print(f"\n  Per-category:")
    for s in category_summaries:
        status = "✓" if s["errors"] == 0 else "⚠"
        print(f"    {status} {s['category']}: {s['ok']}/{s['total']} OK, avg {s['avg_time']:.1f}s")

    # Tier distribution
    all_tiers = {}
    for r in all_results:
        all_tiers[r["tier"]] = all_tiers.get(r["tier"], 0) + 1
    print(f"\n  Tier distribution: {all_tiers}")

    # ── Save results ───────────────────────────────────────────────────────
    output = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(all_results),
            "ok": total_ok,
            "errors": total_err,
            "total_time_seconds": round(total_time, 2),
            "avg_time_seconds": round(avg_time, 2),
            "models_exhausted_final": ModelManager.exhausted_count(),
            "tier_distribution": all_tiers,
        },
        "categories": category_summaries,
        "results": all_results,
    }

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Also save a readable text report
    with open("test_results.txt", "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(f"{'=' * 70}\n")
            f.write(f"Category: {r['category']}\n")
            f.write(f"Question: {r['question']}\n")
            f.write(f"Tier: {r['tier']} | Time: {r['time_seconds']}s | Status: {r['status']}\n")
            f.write(f"Sources: {r['sources']}\n")
            f.write(f"Answer:\n{r['answer']}\n\n")

    print(f"\n  Results saved to: test_results.json, test_results.txt")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_tests()
