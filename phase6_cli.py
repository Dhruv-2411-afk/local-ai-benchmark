"""
phase6_cli.py
─────────────
The user-facing CLI for the smart router assistant.

Features:
  - Interactive chat loop (type questions, get answers)
  - Shows which model was selected and why, every time
  - Colour-coded output by category
  - Session history saved to results/session_log.jsonl
  - Special commands: /history, /stats, /routing, /quit

Usage:
    python phase6_cli.py
    python phase6_cli.py --question "What is binary search?"   # single shot
    python phase6_cli.py --verbose                             # show full scores
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from config import RESULTS_DIR, MODELS
from phase5_router import Router, RouterResult

console = Console()

# Category colours for visual feedback
CATEGORY_COLOURS = {
    "factual":   "cyan",
    "reasoning": "yellow",
    "code":      "green",
    "creative":  "magenta",
    "edge":      "red",
}


# ─── Display helpers ──────────────────────────────────────────────────────────

def print_welcome() -> None:
    console.print(Panel(
        "[bold white]Local AI Router[/bold white]\n"
        "[dim]Automatically selects the best local model for your question[/dim]\n\n"
        "[dim]Commands: /history  /stats  /routing  /quit[/dim]",
        border_style="cyan",
        expand=False,
    ))
    console.print()


def print_result(result: RouterResult, verbose: bool = False) -> None:
    cat    = result.classification.category
    colour = CATEGORY_COLOURS.get(cat, "white")

    # Header bar
    header = (
        f"[{colour}]▶ {cat.upper()}[/{colour}]  "
        f"[dim]→[/dim]  [bold]{result.routing.selected_label}[/bold]  "
        f"[dim]({result.routing.reason})[/dim]"
    )
    console.print(header)
    console.print()

    # Answer
    console.print(result.answer)
    console.print()

    # Metrics footer
    metrics = (
        f"[dim]⏱  {result.total_latency_s:.2f}s total  "
        f"│  {result.tokens_per_second:.1f} tok/s  "
        f"│  {result.completion_tokens} tokens  "
        f"│  classify confidence {result.classification.confidence:.0%}[/dim]"
    )
    console.print(metrics)

    if verbose:
        _print_score_table(result)

    console.print()


def _print_score_table(result: RouterResult) -> None:
    t = Table(box=box.SIMPLE, show_header=True, header_style="dim")
    t.add_column("Model",     style="bold")
    t.add_column("Quality",   justify="right")
    t.add_column("TPS",       justify="right")
    t.add_column("Composite", justify="right")
    t.add_column("Selected",  justify="center")

    for s in result.routing.scores:
        selected = "[green]✓[/green]" if s["model"] == result.routing.selected_model else ""
        t.add_row(
            s["label"],
            f"{s['quality']:.2f}",
            f"{s['tps']:.1f}",
            f"{s['composite']:.3f}",
            selected,
        )
    console.print(t)


# ─── Session stats ────────────────────────────────────────────────────────────

def print_stats(history: list[RouterResult]) -> None:
    if not history:
        console.print("[dim]No questions yet.[/dim]")
        return

    from collections import Counter
    import statistics

    cat_counts  = Counter(r.classification.category        for r in history)
    model_counts= Counter(r.routing.selected_model         for r in history)
    avg_latency = statistics.mean(r.total_latency_s        for r in history)
    avg_tps     = statistics.mean(r.tokens_per_second      for r in history)

    t = Table(title="Session Statistics", box=box.ROUNDED)
    t.add_column("Metric", style="bold cyan")
    t.add_column("Value",  justify="right")

    t.add_row("Questions asked",   str(len(history)))
    t.add_row("Avg total latency", f"{avg_latency:.2f}s")
    t.add_row("Avg tokens/sec",    f"{avg_tps:.1f}")
    t.add_row("─" * 20,            "─" * 10)

    for cat, n in cat_counts.most_common():
        colour = CATEGORY_COLOURS.get(cat, "white")
        t.add_row(f"[{colour}]{cat}[/{colour}]", str(n))

    t.add_row("─" * 20, "─" * 10)
    for model, n in model_counts.most_common():
        label = MODELS.get(model, {}).get("label", model)
        t.add_row(label, str(n))

    console.print(t)


def print_routing_table(router: Router) -> None:
    categories = ["factual", "reasoning", "code", "creative", "edge"]
    t = Table(
        title="Current Routing Table (best model per category)",
        box=box.ROUNDED, show_lines=True
    )
    t.add_column("Category",  style="bold")
    t.add_column("Best Model")
    t.add_column("Quality",   justify="right")
    t.add_column("TPS",       justify="right")
    t.add_column("Score",     justify="right")

    for cat in categories:
        scores = router.routing_table.table.get(cat, [])
        if scores:
            s      = scores[0]
            colour = CATEGORY_COLOURS.get(cat, "white")
            t.add_row(
                f"[{colour}]{cat}[/{colour}]",
                s.label,
                f"{s.quality_mean:.2f}",
                f"{s.tps_mean:.1f}",
                f"{s.composite_score:.3f}",
            )
        else:
            t.add_row(cat, "—", "—", "—", "—")

    console.print(t)


def print_history(history: list[RouterResult]) -> None:
    if not history:
        console.print("[dim]No history yet.[/dim]")
        return
    for i, r in enumerate(history, 1):
        cat    = r.classification.category
        colour = CATEGORY_COLOURS.get(cat, "white")
        console.print(
            f"[dim]{i:2}.[/dim] [{colour}]{cat:10}[/{colour}]  "
            f"[bold]{r.routing.selected_label:14}[/bold]  "
            f"{r.question[:55]}"
        )


# ─── Session logger ───────────────────────────────────────────────────────────

def save_session(history: list[RouterResult]) -> None:
    if not history:
        return
    path = RESULTS_DIR / "session_log.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        for r in history:
            row = r.to_dict()
            row["timestamp"] = datetime.now().isoformat()
            fh.write(json.dumps(row) + "\n")
    console.print(f"[dim]Session saved → {path}[/dim]")


# ─── Main loop ────────────────────────────────────────────────────────────────

def interactive_loop(router: Router, verbose: bool) -> None:
    history: list[RouterResult] = []
    print_welcome()

    while True:
        try:
            raw = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not raw:
            continue

        # Commands
        if raw.lower() in ("/quit", "/exit", "/q"):
            break
        elif raw.lower() == "/history":
            print_history(history)
            continue
        elif raw.lower() == "/stats":
            print_stats(history)
            continue
        elif raw.lower() == "/routing":
            print_routing_table(router)
            continue
        elif raw.lower() == "/help":
            console.print(
                "[dim]/history  — show past questions\n"
                "/stats    — session statistics\n"
                "/routing  — show routing table\n"
                "/quit     — exit[/dim]"
            )
            continue

        # Route and display
        with console.status("[dim]Classifying and routing…[/dim]"):
            result = router.route(raw)

        history.append(result)

        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")
        else:
            print_result(result, verbose=verbose)

    save_session(history)
    console.print("\n[dim]Goodbye.[/dim]")


# ─── CLI entry-point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Local AI Router — CLI")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Ask a single question and exit")
    parser.add_argument("--verbose",  "-v", action="store_true",
                        help="Show full model score breakdown per query")
    args = parser.parse_args()

    console.print("[dim]Initialising router…[/dim]")
    router = Router()

    if args.question:
        # Single-shot mode
        with console.status("[dim]Routing…[/dim]"):
            result = router.route(args.question)
        if result.error:
            console.print(f"[red]Error: {result.error}[/red]")
            sys.exit(1)
        print_result(result, verbose=args.verbose)
    else:
        interactive_loop(router, verbose=args.verbose)


if __name__ == "__main__":
    main()
