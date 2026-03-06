#!/usr/bin/env python3
"""
cli_chat.py — Interactive CLI chat interface for the Legal RAG Chatbot.

Run after building the index:

    python cli_chat.py

Commands:
    /quit, /exit   — Exit the chat
    /clear         — Clear conversation history
    /cls           — Clear screen
    /sources       — Toggle source display (on by default)
    /debug         — Toggle debug info (rewritten query, timings)
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Suppress noisy logs in CLI mode
logging.basicConfig(level=logging.WARNING)


def main() -> None:
    # Rich is optional — fall back to plain print
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        use_rich = True
    except ImportError:
        console = None
        use_rich = False

    def rprint(*args, **kwargs):
        if use_rich:
            console.print(*args, **kwargs)
        else:
            print(*args)

    rprint()
    if use_rich:
        rprint(Panel.fit(
            "[bold cyan]Indian Legal RAG Chatbot v3.0[/bold cyan]\n"
            "[dim]Deep-Research Agentic Pipeline: intent → plan → "
            "iterative retrieval → evidence graph → reasoning → verify[/dim]\n\n"
            "Type your legal question and press Enter.\n"
            "Commands: /quit  /clear  /cls  /sources  /debug",
            title="⚖️  Legal Assistant",
            border_style="cyan",
        ))
    else:
        print("=" * 60)
        print("  Indian Legal RAG Chatbot v3.0")
        print("  Deep-Research Agentic Pipeline")
        print("  Commands: /quit /clear /cls /sources /debug")
        print("=" * 60)

    # ── Initialise chatbot ─────────────────────────────────────────────────
    rprint("\n[dim]Loading pipeline (embedding model + cross-encoder + FAISS + BM25)…[/dim]" if use_rich else "\nLoading pipeline…")
    from chatbot.chatbot import LegalChatbot

    index_dir = os.getenv("INDEX_DIR", "index")
    bot = LegalChatbot(index_dir=index_dir)
    rprint("[green]Ready![/green]\n" if use_rich else "Ready!\n")

    show_sources = True
    show_debug = False

    while True:
        try:
            question = input("You ➤  ").strip()
        except (EOFError, KeyboardInterrupt):
            rprint("\nGoodbye!")
            break

        if not question:
            continue

        # ── Commands ───────────────────────────────────────────────────────
        cmd = question.lower()
        if cmd in ("/quit", "/exit"):
            rprint("Goodbye!")
            break
        if cmd == "/cls":
            os.system("cls" if os.name == "nt" else "clear")
            continue
        if cmd == "/clear":
            bot.clear_history()
            rprint("[yellow]Conversation history cleared.[/yellow]" if use_rich else "History cleared.")
            continue
        if cmd == "/sources":
            show_sources = not show_sources
            rprint(f"[yellow]Source display: {'ON' if show_sources else 'OFF'}[/yellow]" if use_rich else f"Source display: {'ON' if show_sources else 'OFF'}")
            continue
        if cmd == "/debug":
            show_debug = not show_debug
            rprint(f"[yellow]Debug info: {'ON' if show_debug else 'OFF'}[/yellow]" if use_rich else f"Debug: {'ON' if show_debug else 'OFF'}")
            continue

        # ── Ask the chatbot (streaming) ────────────────────────────────────
        rprint()
        if use_rich:
            full_answer = ""
            with console.status("[bold cyan]Thinking…[/bold cyan]"):
                for token in bot.ask_stream(question):
                    full_answer += token
            console.print(Markdown(full_answer))
        else:
            full_answer = ""
            print("Bot: ", end="", flush=True)
            for token in bot.ask_stream(question):
                full_answer += token
                print(token, end="", flush=True)
            print()

        # ── Sources ────────────────────────────────────────────────────────
        _FAILURE_MSG = "I couldn't generate an answer right now"
        if show_sources and _FAILURE_MSG not in full_answer:
            sources = bot.get_last_sources()
            if sources:
                rprint()
                if use_rich:
                    source_text = "\n".join(f"  • {s}" for s in sources)
                    rprint(Panel(source_text, title="📚 Sources", border_style="dim"))
                else:
                    print("Sources:")
                    for s in sources:
                        print(f"  • {s}")

        # ── Debug info ─────────────────────────────────────────────────────
        if show_debug:
            rewritten = bot.get_last_rewritten_query()
            plan = bot.get_last_research_plan()
            graph_stats = bot.get_last_graph_stats()
            complexity = bot.get_last_complexity()
            iterations = bot.get_last_retrieval_iterations()
            follow_ups = bot.get_last_follow_up_queries()

            if use_rich:
                debug_parts: list[str] = []
                if rewritten and rewritten != question:
                    debug_parts.append(f"[bold]Rewritten query:[/bold] {rewritten}")
                if complexity and complexity != "n/a":
                    debug_parts.append(f"[bold]Complexity:[/bold] {complexity}")
                if plan:
                    num_tasks = len(plan.get("research_tasks", []))
                    debug_parts.append(f"[bold]Research tasks:[/bold] {num_tasks}")
                    for t in plan.get("research_tasks", []):
                        debug_parts.append(f"  • Task {t.get('id', '?')}: {t.get('description', '')}")
                if iterations and iterations > 0:
                    debug_parts.append(f"[bold]Retrieval iterations:[/bold] {iterations}")
                if graph_stats:
                    debug_parts.append(
                        f"[bold]Evidence graph:[/bold] {graph_stats.get('nodes', 0)} nodes, "
                        f"{graph_stats.get('edges', 0)} edges, "
                        f"{graph_stats.get('chunks', 0)} chunks"
                    )
                if follow_ups:
                    debug_parts.append(f"[bold]Follow-up queries:[/bold] {', '.join(follow_ups)}")
                if debug_parts:
                    rprint()
                    rprint(Panel(
                        "\n".join(debug_parts),
                        title="🔬 Deep Research Debug",
                        border_style="dim blue",
                    ))
            else:
                if rewritten and rewritten != question:
                    print(f"Rewritten query: {rewritten}")
                if complexity and complexity != "n/a":
                    print(f"Complexity: {complexity}")
                if plan:
                    print(f"Research tasks: {len(plan.get('research_tasks', []))}")
                if iterations and iterations > 0:
                    print(f"Retrieval iterations: {iterations}")
                if graph_stats:
                    print(f"Evidence graph: {graph_stats}")
                if follow_ups:
                    print(f"Follow-up queries: {follow_ups}")

        rprint()


if __name__ == "__main__":
    main()
