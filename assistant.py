"""
assistant.py
────────────
Smart local AI assistant with conversation memory and model routing.

Features:
  - Remembers last 5 exchanges (conversation memory)
  - Routes each question to the best model using benchmark data
  - Shows which model was chosen and why
  - --speed flag to prefer fastest model over best quality
  - --explain flag to show routing decisions
  - /history, /models, /stats, /clear, /exit commands

Usage:
    python assistant.py                  # quality-first routing
    python assistant.py --speed          # speed-first routing
    python assistant.py --explain        # show routing decisions
"""

import time
import argparse
import statistics
from datetime import datetime
from collections import defaultdict

import ollama
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from config import MODELS, log
from router import Router, RoutedResponse, RoutingDecision
from classifier import classify, ClassificationResult

console = Console()

MEMORY_WINDOW = 5

CATEGORY_COLORS = {
    "factual":   "cyan",
    "reasoning": "yellow",
    "code":      "green",
    "creative":  "magenta",
    "edge":      "red",
}

MODEL_COLORS = {
    "llama3.2:3b": "blue",
    "phi4-mini":   "green",
    "mistral:7b":  "yellow",
}


def print_welcome(prefer_speed: bool) -> None:
    mode = "[yellow]SPEED[/yellow]" if prefer_speed else "[green]QUALITY[/green]"
    console.print(Panel(
        f"[bold]Local AI Router[/bold]\n"
        f"Routing mode: {mode}\n"
        f"Memory: last {MEMORY_WINDOW} exchanges\n\n"
        f"[dim]Commands: /history  /models  /stats  /clear  /exit[/dim]\n"
        f"[dim]Ask anything — the best model is chosen automatically.[/dim]",
        title="[bold cyan]Smart Local Assistant[/bold cyan]",
        border_style="cyan",
    ))


def print_routing_badge(response: RoutedResponse) -> None:
    cat    = response.category
    model  = response.model_used
    label  = MODELS.get(model, {}).get("label", model)
    color  = CATEGORY_COLORS.get(cat, "white")
    mcolor = MODEL_COLORS.get(model, "white")
    console.print(
        f"  [dim]-> classified as[/dim] [{color}]{cat}[/{color}]  "
        f"[dim]-> routed to[/dim] [{mcolor}]{label}[/{mcolor}]  "
        f"[dim]({response.latency_s:.2f}s | {response.tokens_per_sec:.0f} tok/s)[/dim]"
    )


def print_answer(response: RoutedResponse) -> None:
    print_routing_badge(response)
    console.print()
    console.print(response.answer)
    console.print()


def print_routing_detail(response: RoutedResponse) -> None:
    d = response.routing
    c = response.classification
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Key",   style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Question category",     c.category)
    table.add_row("Classifier confidence", f"{c.confidence:.0%}")
    table.add_row("Classifier reason",     c.reason)
    table.add_row("Chosen model",          MODELS.get(d.chosen_model, {}).get("label", d.chosen_model))
    table.add_row("Routing reason",        d.reason)
    table.add_row("Quality score",         f"{d.quality_score:.2f} / 3.00")
    table.add_row("Model TPS",             f"{d.tps:.1f}")
    if d.alternatives:
        alts = ", ".join(
            f"{MODELS.get(a['model'],{}).get('label', a['model'])} (Q={a['quality']:.2f})"
            for a in d.alternatives
        )
        table.add_row("Alternatives", alts)
    console.print(Panel(table, title="[bold]Routing Decision[/bold]", border_style="dim"))


class Session:
    def __init__(self):
        self.history:    list[RoutedResponse] = []
        self.started_at: datetime             = datetime.now()

    def add(self, r: RoutedResponse) -> None:
        self.history.append(r)

    def print_stats(self) -> None:
        if not self.history:
            console.print("[dim]No queries yet.[/dim]")
            return
        by_model: dict[str, list] = defaultdict(list)
        by_cat:   dict[str, int]  = defaultdict(int)
        for r in self.history:
            by_model[r.model_used].append(r)
            by_cat[r.category] += 1
        console.print(Panel(
            f"[bold]Session Stats[/bold]  "
            f"[dim]{len(self.history)} queries | "
            f"{(datetime.now() - self.started_at).seconds}s elapsed[/dim]",
            border_style="dim",
        ))
        t = Table(box=box.SIMPLE)
        t.add_column("Model",   style="bold")
        t.add_column("Queries", justify="right")
        t.add_column("Avg TPS", justify="right")
        t.add_column("Avg Lat", justify="right")
        for model, recs in by_model.items():
            label = MODELS.get(model, {}).get("label", model)
            t.add_row(
                label,
                str(len(recs)),
                f"{statistics.mean(r.tokens_per_sec for r in recs):.1f}",
                f"{statistics.mean(r.latency_s      for r in recs):.2f}s",
            )
        console.print(t)
        console.print(
            "[bold]By category:[/bold] " +
            "  ".join(
                f"[{CATEGORY_COLORS.get(c, 'white')}]{c}[/] x{n}"
                for c, n in sorted(by_cat.items())
            )
        )
        console.print()

    def print_history(self) -> None:
        if not self.history:
            console.print("[dim]No history yet.[/dim]")
            return
        for i, r in enumerate(self.history, 1):
            label = MODELS.get(r.model_used, {}).get("label", r.model_used)
            color = CATEGORY_COLORS.get(r.category, "white")
            console.print(
                f"  [dim]{i:2}.[/dim] [{color}]{r.category:10}[/{color}]  "
                f"[dim]{label:15}[/dim]  {r.question[:60]}"
            )
        console.print()

    def print_models(self) -> None:
        t = Table(title="Available Models", box=box.ROUNDED)
        t.add_column("Model",  style="bold")
        t.add_column("Params", justify="right")
        t.add_column("Tag")
        for tag, info in MODELS.items():
            t.add_row(info["label"], f"{info['params_b']}B", tag)
        console.print(t)


def route_with_memory(
    router: Router,
    question: str,
    memory: list[dict],
) -> RoutedResponse:
    classification = classify(question)
    from router import pick_model
    decision = pick_model(
        classification.category,
        router.routing_table,
        prefer_speed=router.prefer_speed,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful local AI assistant. "
                "You have memory of the recent conversation. "
                "Be concise and accurate."
            ),
        }
    ]
    messages.extend(memory[-(MEMORY_WINDOW * 2):])
    messages.append({"role": "user", "content": question})

    client = ollama.Client()
    t0 = time.perf_counter()

    try:
        resp = client.chat(
            model=decision.chosen_model,
            messages=messages,
            options={"temperature": 0.7},
            stream=False,
        )
        elapsed = time.perf_counter() - t0
        answer  = resp["message"]["content"]
        tokens  = resp.get("eval_count", 0)
        tps     = tokens / elapsed if elapsed > 0 else 0.0

        return RoutedResponse(
            question=question,
            answer=answer,
            model_used=decision.chosen_model,
            category=classification.category,
            classification=classification,
            routing=decision,
            latency_s=round(elapsed, 3),
            tokens=tokens,
            tokens_per_sec=round(tps, 1),
        )

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        log.error("Model call failed: %s", exc)
        return RoutedResponse(
            question=question,
            answer="",
            model_used=decision.chosen_model,
            category=classification.category,
            classification=classification,
            routing=decision,
            latency_s=round(elapsed, 3),
            tokens=0,
            tokens_per_sec=0.0,
            error=str(exc),
        )


def chat_loop(
    router: Router,
    explain_mode: bool = False,
    session: Session = None,
) -> None:
    if session is None:
        session = Session()

    memory: list[dict] = []

    while True:
        try:
            console.print("[bold cyan]You:[/bold cyan] ", end="")
            question = input().strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue

        if question.lower() in ("/exit", "/quit", "exit", "quit"):
            break
        if question.lower() == "/stats":
            session.print_stats()
            continue
        if question.lower() == "/history":
            session.print_history()
            continue
        if question.lower() == "/models":
            session.print_models()
            continue
        if question.lower() == "/clear":
            memory.clear()
            console.print("[dim]Memory cleared.[/dim]\n")
            continue

        with console.status("[dim]Thinking...[/dim]"):
            response = route_with_memory(router, question, memory)

        session.add(response)

        if response.error:
            console.print(f"[red]Error: {response.error}[/red]\n")
            continue

        memory.append({"role": "user",      "content": question})
        memory.append({"role": "assistant", "content": response.answer})

        exchanges = len(memory) // 2
        mem_label = f"[dim](memory: {exchanges}/{MEMORY_WINDOW} exchanges)[/dim]"

        console.print(f"\n[bold green]Assistant:[/bold green] {mem_label}")

        if explain_mode:
            print_routing_detail(response)

        print_answer(response)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Local AI Assistant")
    parser.add_argument("--speed",   action="store_true",
                        help="Prefer fastest model over best quality")
    parser.add_argument("--explain", action="store_true",
                        help="Show detailed routing decision for each query")
    parser.add_argument("--query",   type=str, default=None,
                        help="Single query mode (non-interactive)")
    args = parser.parse_args()

    router  = Router(prefer_speed=args.speed)
    session = Session()

    print_welcome(args.speed)

    if args.query:
        response = router.route(args.query)
        if args.explain:
            print_routing_detail(response)
        print_answer(response)
        return

    try:
        chat_loop(router, explain_mode=args.explain, session=session)
    finally:
        console.print("\n[bold]Session summary:[/bold]")
        session.print_stats()
        console.print("[dim]Goodbye.[/dim]\n")


if __name__ == "__main__":
    main()
