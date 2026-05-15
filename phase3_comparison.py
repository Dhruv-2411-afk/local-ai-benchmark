"""
phase3_comparison.py
────────────────────
Comparative benchmark across three models:
  - llama3.2:3b  (3.2 B parameters)
  - phi4-mini    (3.8 B parameters)
  - mistral:7b   (7.0 B parameters)

Metrics collected per (model, prompt):
  ┌─────────────────────────────────────────────────────────┐
  │  Performance   │ tokens/sec, TTFT, total latency        │
  │  Memory        │ RSS before & after, peak delta (psutil)│
  │  Output quality│ length, schema validity, self-rated    │
  │                │ confidence, manual quality score        │
  └─────────────────────────────────────────────────────────┘

Quality scoring (automated, 0-3 per prompt):
  +1  response is non-empty and longer than 10 words
  +1  contains expected keywords for factual / code prompts
  +1  passes Pydantic AssistantResponse schema

Usage:
    python phase3_comparison.py [--models llama3.2:3b phi4-mini mistral:7b]
                                [--prompts 40] [--runs 1]
"""

import time
import json
import argparse
import statistics
import psutil
import os
import gc
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Optional

import ollama
from rich.console import Console
from rich.table import Table
from rich import box

from config import MODELS, RESULTS_DIR, save_jsonl, log
from prompts import PROMPTS, PROMPT_BY_ID
from phase1_measurement import measure_single
from phase2_structured import validate_response, SYSTEM_PROMPT

console = Console()
_proc   = psutil.Process(os.getpid())


# ════════════════════════════════════════════════════════════════════════════
# Data model
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ComparisonRecord:
    model:               str
    prompt_id:           str
    category:            str
    run:                 int
    # performance
    ttft_s:              float
    total_latency_s:     float
    tokens_per_second:   float
    prompt_tokens:       int
    completion_tokens:   int
    # memory
    rss_before_mb:       float
    rss_after_mb:        float
    rss_delta_mb:        float
    # quality
    response_text:       str
    response_words:      int
    schema_valid:        bool
    quality_score:       int          # 0-3
    error:               Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ════════════════════════════════════════════════════════════════════════════
# Quality scorer (fully automated, no human loop)
# ════════════════════════════════════════════════════════════════════════════

# Simple keyword heuristics per factual prompt id
FACTUAL_KEYWORDS: dict[str, list[str]] = {
    "f01": ["canberra"],
    "f02": ["1989"],
    "f03": ["299", "792", "458"],
    "f04": ["gravitational", "electromagnetic", "weak", "strong"],
    "f05": ["monty", "python", "circus"],
    "f06": ["206"],
    "f07": ["hypertext", "transfer", "protocol"],
    "f08": ["austen", "jane"],
}

def auto_quality_score(record: "ComparisonRecord", prompt: dict) -> int:
    """
    Returns 0-3 based on automated checks:
      0 — empty / error
      1 — non-trivially long (>10 words)
      2 — contains expected keywords (factual) or code fence (code)
      3 — also passes Pydantic schema validation
    """
    score = 0
    text  = (record.response_text or "").lower()

    if record.error or not text.strip():
        return 0

    words = text.split()
    if len(words) >= 10:
        score = 1

    pid = prompt["id"]
    cat = prompt["category"]

    if cat == "factual" and pid in FACTUAL_KEYWORDS:
        if any(kw in text for kw in FACTUAL_KEYWORDS[pid]):
            score = max(score, 2)
        else:
            score = max(score, 1)   # answered but possibly wrong
    elif cat == "code":
        if "def " in text or "```" in text or "SELECT" in text.upper():
            score = max(score, 2)
    elif cat in ("reasoning", "creative", "edge"):
        # give benefit-of-doubt for non-trivially long answers
        if len(words) >= 20:
            score = max(score, 2)

    # Bonus: schema round-trip passes
    _, err = validate_response(record.response_text)
    if err is None:
        score = 3

    return score


# ════════════════════════════════════════════════════════════════════════════
# Memory-aware inference call
# ════════════════════════════════════════════════════════════════════════════

def rss_mb() -> float:
    return _proc.memory_info().rss / (1024 ** 2)


def run_one(
    client: ollama.Client,
    model_name: str,
    prompt: dict,
    run: int,
) -> ComparisonRecord:
    """One timed + memory-tracked inference call."""
    gc.collect()
    mb_before = rss_mb()

    lat = measure_single(model_name, prompt, temperature=0.0)

    mb_after = rss_mb()
    schema_ok, _ = validate_response(lat.response_text)

    rec = ComparisonRecord(
        model              = model_name,
        prompt_id          = prompt["id"],
        category           = prompt["category"],
        run                = run,
        ttft_s             = lat.time_to_first_token,
        total_latency_s    = lat.total_latency,
        tokens_per_second  = lat.tokens_per_second,
        prompt_tokens      = lat.prompt_tokens,
        completion_tokens  = lat.completion_tokens,
        rss_before_mb      = round(mb_before, 1),
        rss_after_mb       = round(mb_after,  1),
        rss_delta_mb       = round(mb_after - mb_before, 1),
        response_text      = lat.response_text,
        response_words     = len(lat.response_text.split()),
        schema_valid       = schema_ok is not None,
        quality_score      = 0,           # filled below
        error              = lat.error,
    )
    rec.quality_score = auto_quality_score(rec, prompt)
    return rec


# ════════════════════════════════════════════════════════════════════════════
# Benchmark runner
# ════════════════════════════════════════════════════════════════════════════

def run_model_benchmark(
    model_name: str,
    prompts: list[dict],
    runs: int = 1,
) -> list[ComparisonRecord]:
    client  = ollama.Client()
    records = []
    total   = len(prompts) * runs

    with console.status(f"[bold cyan]Benchmarking {model_name}…") as status:
        done = 0
        for prompt in prompts:
            for run in range(1, runs + 1):
                done += 1
                status.update(f"[cyan]{model_name} | {prompt['id']} run {run} [{done}/{total}]")
                rec = run_one(client, model_name, prompt, run)
                log.info(
                    "%-12s | %-5s | %5.1f tok/s | ΔRSS %+.1f MB | Q=%d",
                    model_name, prompt["id"],
                    rec.tokens_per_second, rec.rss_delta_mb, rec.quality_score,
                )
                records.append(rec)

    return records


# ════════════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ════════════════════════════════════════════════════════════════════════════

def aggregate(records: list[ComparisonRecord]) -> dict:
    ok = [r for r in records if not r.error]
    if not ok:
        return {}

    def avg(vals): return round(statistics.mean(vals), 2) if vals else 0

    return {
        "n":               len(ok),
        "tps_mean":        avg([r.tokens_per_second  for r in ok]),
        "tps_median":      round(statistics.median([r.tokens_per_second for r in ok]), 2),
        "ttft_mean_s":     avg([r.ttft_s             for r in ok]),
        "latency_mean_s":  avg([r.total_latency_s    for r in ok]),
        "rss_delta_mean":  avg([r.rss_delta_mb       for r in ok]),
        "quality_mean":    avg([r.quality_score       for r in ok]),
        "quality_max":     max(r.quality_score        for r in ok),
        "schema_pct":      round(100 * sum(r.schema_valid for r in ok) / len(ok), 1),
        "words_mean":      avg([r.response_words      for r in ok]),
        "errors":          len(records) - len(ok),
    }


# ════════════════════════════════════════════════════════════════════════════
# Rich comparison table
# ════════════════════════════════════════════════════════════════════════════

def print_comparison(all_records: dict[str, list[ComparisonRecord]]) -> None:
    aggs = {m: aggregate(r) for m, r in all_records.items()}

    # ── Overall comparison ────────────────────────────────────────────────
    t = Table(
        title="Phase 3 — Model Comparison Overview",
        box=box.DOUBLE_EDGE, show_lines=True,
    )
    t.add_column("Metric",          style="bold cyan", width=26)
    for m in aggs:
        label = MODELS.get(m, {}).get("label", m)
        params = MODELS.get(m, {}).get("params_b", "?")
        t.add_column(f"{label}\n({params}B)", justify="right")

    rows = [
        ("Tokens / Second (mean)",    "tps_mean",       ""),
        ("Tokens / Second (median)",  "tps_median",     ""),
        ("Time-to-First-Token (s)",   "ttft_mean_s",    "s"),
        ("Total Latency (s)",         "latency_mean_s", "s"),
        ("RAM Delta (MB)",            "rss_delta_mean", "MB"),
        ("Quality Score (0-3)",       "quality_mean",   ""),
        ("Schema Valid %",            "schema_pct",     "%"),
        ("Avg Response Words",        "words_mean",     ""),
        ("Errors",                    "errors",         ""),
    ]

    for label, key, unit in rows:
        vals = []
        for m in aggs:
            v = aggs[m].get(key, "N/A")
            vals.append(f"{v}{unit}" if v != "N/A" else "N/A")
        t.add_row(label, *vals)

    console.print()
    console.print(t)

    # ── Per-category quality ──────────────────────────────────────────────
    cat_table = Table(
        title="Quality by Category (mean score 0-3)",
        box=box.SIMPLE, show_lines=True
    )
    cat_table.add_column("Category", style="bold")
    for m in aggs:
        cat_table.add_column(MODELS.get(m, {}).get("label", m), justify="right")

    categories = sorted({r.category for recs in all_records.values() for r in recs})
    for cat in categories:
        row = [cat]
        for model_name, recs in all_records.items():
            cat_recs = [r for r in recs if r.category == cat]
            if cat_recs:
                avg_q = statistics.mean(r.quality_score for r in cat_recs)
                row.append(f"{avg_q:.2f}")
            else:
                row.append("—")
        cat_table.add_row(*row)

    console.print()
    console.print(cat_table)


def print_winner_analysis(all_records: dict[str, list[ComparisonRecord]]) -> None:
    aggs = {m: aggregate(r) for m, r in all_records.items()}

    console.print("\n[bold underline]Phase 3 — Winner Analysis[/bold underline]\n")

    def best_by(key: str, lower_is_better=False) -> str:
        vals = {m: aggs[m].get(key, float("inf") if lower_is_better else 0)
                for m in aggs if aggs[m]}
        if lower_is_better:
            winner = min(vals, key=vals.get)
        else:
            winner = max(vals, key=vals.get)
        label = MODELS.get(winner, {}).get("label", winner)
        return f"[green]{label}[/green] ({vals[winner]:.2f})"

    console.print(f"  Fastest (TPS):         {best_by('tps_mean')}")
    console.print(f"  Lowest Latency:        {best_by('latency_mean_s', lower_is_better=True)}")
    console.print(f"  Lowest Memory Impact:  {best_by('rss_delta_mean', lower_is_better=True)}")
    console.print(f"  Best Quality Score:    {best_by('quality_mean')}")
    console.print(f"  Best Schema Adherence: {best_by('schema_pct')}")


# ════════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — Model Comparison")
    parser.add_argument(
        "--models", nargs="+",
        default=list(MODELS.keys()),
        help="Models to compare (must be pulled in Ollama)",
    )
    parser.add_argument("--prompts", type=int, default=len(PROMPTS),
                        help="Number of prompts from the bank to use")
    parser.add_argument("--runs",    type=int, default=1,
                        help="Repetitions per (model, prompt) pair")
    args = parser.parse_args()

    subset = PROMPTS[: args.prompts]
    console.rule(
        f"[bold]Phase 3 — Comparing {len(args.models)} models  "
        f"× {len(subset)} prompts × {args.runs} run(s)[/bold]"
    )

    all_records: dict[str, list[ComparisonRecord]] = {}

    for model_name in args.models:
        console.rule(f"[yellow]{model_name}[/yellow]")
        records = run_model_benchmark(model_name, subset, runs=args.runs)
        all_records[model_name] = records

        out = RESULTS_DIR / f"phase3_{model_name.replace(':', '_').replace('/', '_')}.jsonl"
        save_jsonl([r.to_dict() for r in records], out)

    print_comparison(all_records)
    print_winner_analysis(all_records)

    # Save aggregates
    aggs = {m: aggregate(r) for m, r in all_records.items()}
    agg_path = RESULTS_DIR / "phase3_aggregates.json"
    agg_path.write_text(json.dumps(aggs, indent=2))
    console.print(f"\n[green]Aggregates saved → {agg_path}[/green]")


if __name__ == "__main__":
    main()
