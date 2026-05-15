"""
phase1_measurement.py
─────────────────────
Measures three core performance metrics for a local Ollama model:

  1. Time-to-First-Token (TTFT)  — latency until the first byte streams back
  2. Total Response Latency      — wall-clock time for the full response
  3. Tokens per Second (TPS)     — completion_tokens / generation_time

Usage:
    python phase1_measurement.py [--model llama3.2:3b] [--runs 3]

Results are written to results/phase1_results.jsonl and a summary table
is printed to stdout via Rich.
"""

import time
import argparse
import statistics
from collections import defaultdict

import ollama
from rich.console import Console
from rich.table import Table
from rich import box

from config import (
    DEFAULT_MODEL, RESULTS_DIR, LatencyRecord,
    save_jsonl, ms, log
)
from prompts import PROMPTS

console = Console()

# ── Measurement core ──────────────────────────────────────────────────────────

def measure_single(
    model: str,
    prompt: dict,
    temperature: float = 0.0,
) -> LatencyRecord:
    """
    Run one prompt through Ollama's streaming API and capture:
      - time_to_first_token  (TTFT)
      - total_latency        (wall-clock generation time)
      - tokens_per_second    (completion_tokens / generation_time)

    Streaming is mandatory — without it, TTFT cannot be measured.
    """
    client = ollama.Client()
    t_start = time.perf_counter()
    t_first  = None

    full_text        = []
    prompt_tokens    = 0
    completion_tokens = 0

    try:
        stream = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt["text"]}],
            options={"temperature": temperature, "seed": 42},
            stream=True,
        )

        for chunk in stream:
            if t_first is None:
                t_first = time.perf_counter()

            content = chunk["message"]["content"]
            full_text.append(content)

            # Ollama surfaces token counts only on the final chunk
            if chunk.get("done"):
                prompt_tokens     = chunk.get("prompt_eval_count", 0)
                completion_tokens = chunk.get("eval_count", 0)

        t_end = time.perf_counter()

        ttft             = (t_first or t_end) - t_start
        total_latency    = t_end - t_start
        generation_time  = total_latency - ttft          # exclude prefill
        tps              = (
            completion_tokens / generation_time
            if generation_time > 0 else 0.0
        )

        return LatencyRecord(
            model=model,
            prompt_id=prompt["id"],
            prompt_text=prompt["text"],
            temperature=temperature,
            time_to_first_token=round(ttft, 4),
            total_latency=round(total_latency, 4),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_per_second=round(tps, 2),
            response_text="".join(full_text),
            phase="phase1",
        )

    except Exception as exc:
        t_end = time.perf_counter()
        log.error("Inference error for prompt %s: %s", prompt["id"], exc)
        return LatencyRecord(
            model=model,
            prompt_id=prompt["id"],
            prompt_text=prompt["text"],
            temperature=temperature,
            time_to_first_token=0.0,
            total_latency=round(t_end - t_start, 4),
            prompt_tokens=0,
            completion_tokens=0,
            tokens_per_second=0.0,
            response_text="",
            error=str(exc),
            phase="phase1",
        )


# ── Benchmark runner ──────────────────────────────────────────────────────────

def run_benchmark(
    model: str,
    prompts: list[dict],
    runs: int = 3,
    temperature: float = 0.0,
) -> list[LatencyRecord]:
    """
    For each prompt, run `runs` repetitions and keep all records.
    Multiple runs expose variance / cold-vs-warm cache effects.
    """
    records: list[LatencyRecord] = []

    with console.status(f"[bold cyan]Benchmarking {model} …") as status:
        for i, prompt in enumerate(prompts, 1):
            for run in range(1, runs + 1):
                status.update(
                    f"[bold cyan]{model}  prompt {i}/{len(prompts)}  run {run}/{runs}"
                )
                rec = measure_single(model, prompt, temperature)
                records.append(rec)
                log.info(
                    "%-6s | %-5s | TTFT %s | latency %s | %5.1f tok/s",
                    prompt["id"], f"run{run}",
                    ms(rec.time_to_first_token),
                    ms(rec.total_latency),
                    rec.tokens_per_second,
                )

    return records


# ── Summary statistics ────────────────────────────────────────────────────────

def compute_summary(records: list[LatencyRecord]) -> dict:
    """Aggregate mean / median / p95 across all successful records."""
    ok = [r for r in records if not r.error]
    if not ok:
        return {}

    def stats(values: list[float]) -> dict:
        values = sorted(values)
        p95_idx = max(0, int(len(values) * 0.95) - 1)
        return {
            "mean":   round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "stdev":  round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            "p95":    round(values[p95_idx], 3),
            "min":    round(min(values), 3),
            "max":    round(max(values), 3),
        }

    return {
        "n_samples":             len(ok),
        "time_to_first_token_s": stats([r.time_to_first_token for r in ok]),
        "total_latency_s":       stats([r.total_latency       for r in ok]),
        "tokens_per_second":     stats([r.tokens_per_second   for r in ok]),
        "completion_tokens":     stats([r.completion_tokens   for r in ok]),
    }


def print_summary_table(model: str, summary: dict) -> None:
    table = Table(
        title=f"[bold]Phase 1 — {model}[/bold]",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Metric",  style="bold cyan", width=30)
    table.add_column("Mean",    justify="right")
    table.add_column("Median",  justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("P95",     justify="right")
    table.add_column("Min",     justify="right")
    table.add_column("Max",     justify="right")

    def row(label, key, unit=""):
        s = summary.get(key, {})
        if not s:
            return
        fmt = lambda v: f"{v}{unit}"
        table.add_row(
            label,
            fmt(s["mean"]),
            fmt(s["median"]),
            fmt(s["stdev"]),
            fmt(s["p95"]),
            fmt(s["min"]),
            fmt(s["max"]),
        )

    row("Time-to-First-Token (s)", "time_to_first_token_s", "s")
    row("Total Latency (s)",       "total_latency_s",       "s")
    row("Tokens / Second",         "tokens_per_second",     "")
    row("Completion Tokens",       "completion_tokens",      "")

    console.print()
    console.print(table)
    console.print(f"  [dim]n = {summary['n_samples']} samples[/dim]\n")


# ── Category breakdown ────────────────────────────────────────────────────────

def print_category_breakdown(records: list[LatencyRecord]) -> None:
    """Show per-category averages to expose prompt-length effects."""
    from prompts import PROMPT_BY_ID

    by_cat: dict[str, list] = defaultdict(list)
    for r in records:
        if r.error:
            continue
        cat = PROMPT_BY_ID[r.prompt_id]["category"]
        by_cat[cat].append(r)

    table = Table(title="Phase 1 — Category Breakdown", box=box.SIMPLE)
    table.add_column("Category",  style="bold")
    table.add_column("n",         justify="right")
    table.add_column("Avg TTFT",  justify="right")
    table.add_column("Avg TPS",   justify="right")
    table.add_column("Avg Tok",   justify="right")

    for cat in sorted(by_cat):
        recs = by_cat[cat]
        table.add_row(
            cat,
            str(len(recs)),
            f"{statistics.mean(r.time_to_first_token for r in recs):.3f}s",
            f"{statistics.mean(r.tokens_per_second   for r in recs):.1f}",
            f"{statistics.mean(r.completion_tokens   for r in recs):.0f}",
        )

    console.print(table)


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 — Latency benchmark")
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--runs",   type=int, default=3,
                        help="Repetitions per prompt (default 3)")
    parser.add_argument("--prompts", type=int, default=len(PROMPTS),
                        help="How many prompts to use (default: all 40)")
    args = parser.parse_args()

    subset = PROMPTS[: args.prompts]

    console.rule(f"[bold]Phase 1 — {args.model}  ({args.runs} runs × {len(subset)} prompts)[/bold]")

    records  = run_benchmark(args.model, subset, runs=args.runs)
    summary  = compute_summary(records)

    print_summary_table(args.model, summary)
    print_category_breakdown(records)

    # Persist
    out_path = RESULTS_DIR / "phase1_results.jsonl"
    save_jsonl([r.to_dict() for r in records], out_path)

    import json
    sum_path = RESULTS_DIR / "phase1_summary.json"
    sum_path.write_text(json.dumps({args.model: summary}, indent=2))
    console.print(f"[green]Results saved → {out_path}[/green]")


if __name__ == "__main__":
    main()
