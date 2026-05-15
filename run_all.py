"""
run_all.py
──────────
Master runner — executes all three phases in sequence and generates the
technical report. Designed for a single unattended run on a new machine.

Usage:
    # Full run (all 40 prompts, all models, 3 reps for Phase 1)
    python run_all.py

    # Quick smoke-test (5 prompts, llama only, 1 rep)
    python run_all.py --quick

    # Skip to report generation (results already exist)
    python run_all.py --report-only
"""

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def banner(text: str) -> None:
    console.print(Panel(f"[bold yellow]{text}[/bold yellow]", expand=False))


def run(cmd: list[str]) -> None:
    console.print(f"[dim]$ {' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        console.print(f"[red]Command exited with code {result.returncode}[/red]")


def pull_models(models: list[str]) -> None:
    """Pull models that are not yet available locally."""
    banner("Pulling models from Ollama registry …")
    for m in models:
        run(["ollama", "pull", m])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all benchmark phases")
    parser.add_argument("--quick",       action="store_true",
                        help="Smoke-test: 5 prompts, 1 run, llama only")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip benchmarks, only generate report")
    parser.add_argument("--no-pull",     action="store_true",
                        help="Skip ollama pull (models already local)")
    args = parser.parse_args()

    MODELS = ["llama3.2:3b", "phi4-mini", "mistral:7b"]
    p3_models = MODELS if not args.quick else ["llama3.2:3b"]
    prompt_n  = 5 if args.quick else 40
    p1_runs   = 1 if args.quick else 3

    if not args.report_only:
        # ── Pull ─────────────────────────────────────────────────────────────
        if not args.no_pull:
            pull_models(MODELS)

        # ── Phase 1 ──────────────────────────────────────────────────────────
        banner("Phase 1 — Latency & Throughput Measurement")
        run([
            sys.executable, "phase1_measurement.py",
            "--model", "llama3.2:3b",
            "--runs",  str(p1_runs),
            "--prompts", str(prompt_n),
        ])

        # ── Phase 2 ──────────────────────────────────────────────────────────
        banner("Phase 2 — Structured Output & Temperature Study")
        temps = ["0.0", "0.7", "1.4"] if args.quick else ["0.0", "0.3", "0.7", "1.0", "1.4"]
        run([
            sys.executable, "phase2_structured.py",
            "--model",   "llama3.2:3b",
            "--temps",   *temps,
            "--prompts", str(prompt_n),
        ])

        # ── Phase 3 ──────────────────────────────────────────────────────────
        banner("Phase 3 — Multi-Model Comparison")
        run([
            sys.executable, "phase3_comparison.py",
            "--models",  *p3_models,
            "--prompts", str(prompt_n),
            "--runs",    "1",
        ])

    # ── Report ───────────────────────────────────────────────────────────────
    banner("Generating Technical Report")
    run([sys.executable, "report_generator.py", "--from-results"])

    console.print("\n[bold green]✓ All done![/bold green]")
    console.print("  Report → results/technical_report.md")
    console.print("  Raw    → results/*.jsonl\n")


if __name__ == "__main__":
    main()
