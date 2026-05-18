"""
router.py
─────────
The core of Option C — uses your Phase 3 benchmark data to route
each question to the best model for that category.

Routing logic:
  1. Classify the question (via classifier.py)
  2. Load Phase 3 quality scores per (model, category)
  3. Pick the model with the highest quality score for that category
  4. Tiebreak on tokens/second (prefer faster model if quality is equal)
  5. Call that model and return a RoutedResponse with full metadata

The benchmark data IS the routing brain — this is what makes the
project unique. You measured it, now you're using it.
"""

import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import defaultdict

import ollama

from config import MODELS, RESULTS_DIR, log
from classifier import classify, ClassificationResult

# ── Routing decision dataclass ────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    chosen_model:    str
    category:        str
    reason:          str
    quality_score:   float
    tps:             float
    alternatives:    list[dict]   # other models considered


@dataclass
class RoutedResponse:
    question:        str
    answer:          str
    model_used:      str
    category:        str
    classification:  ClassificationResult
    routing:         RoutingDecision
    latency_s:       float
    tokens:          int
    tokens_per_sec:  float
    error:           Optional[str] = None


# ── Benchmark data loader ─────────────────────────────────────────────────────

def load_benchmark_scores() -> dict[str, dict[str, dict]]:
    """
    Load Phase 3 results and build a routing table:
      { category → { model → { quality, tps, latency } } }

    Falls back to hardcoded defaults if results don't exist yet.
    """
    routing_table: dict[str, dict[str, dict]] = defaultdict(dict)

    # Try loading from Phase 3 per-model files
    found_data = False
    for model_key in MODELS:
        safe_name = model_key.replace(":", "_").replace("/", "_")
        path = RESULTS_DIR / f"phase3_{safe_name}.jsonl"
        if not path.exists():
            continue

        found_data = True
        by_category: dict[str, list] = defaultdict(list)

        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("error"):
                    by_category[r["category"]].append(r)

        for cat, records in by_category.items():
            avg_q   = sum(r["quality_score"]     for r in records) / len(records)
            avg_tps = sum(r["tokens_per_second"] for r in records) / len(records)
            avg_lat = sum(r["total_latency_s"]   for r in records) / len(records)
            routing_table[cat][model_key] = {
                "quality": round(avg_q,   3),
                "tps":     round(avg_tps, 2),
                "latency": round(avg_lat, 3),
            }

    if found_data:
        log.info("Loaded routing table from Phase 3 benchmark data")
        return dict(routing_table)

    # ── Hardcoded fallback (sensible defaults before full benchmark runs) ──────
    log.warning("No Phase 3 data found — using default routing table")
    defaults = {
        "factual":   {"llama3.2:3b": {"quality": 2.0, "tps": 35.0, "latency": 1.5},
                      "phi4-mini":   {"quality": 2.2, "tps": 28.0, "latency": 2.0},
                      "mistral:7b":  {"quality": 2.5, "tps": 18.0, "latency": 3.5}},
        "reasoning": {"llama3.2:3b": {"quality": 1.5, "tps": 35.0, "latency": 2.5},
                      "phi4-mini":   {"quality": 2.3, "tps": 28.0, "latency": 3.0},
                      "mistral:7b":  {"quality": 2.8, "tps": 18.0, "latency": 5.0}},
        "code":      {"llama3.2:3b": {"quality": 1.8, "tps": 35.0, "latency": 3.0},
                      "phi4-mini":   {"quality": 2.0, "tps": 28.0, "latency": 3.5},
                      "mistral:7b":  {"quality": 2.9, "tps": 18.0, "latency": 6.0}},
        "creative":  {"llama3.2:3b": {"quality": 2.1, "tps": 35.0, "latency": 2.0},
                      "phi4-mini":   {"quality": 1.9, "tps": 28.0, "latency": 2.5},
                      "mistral:7b":  {"quality": 2.3, "tps": 18.0, "latency": 4.0}},
        "edge":      {"llama3.2:3b": {"quality": 2.0, "tps": 35.0, "latency": 1.0},
                      "phi4-mini":   {"quality": 1.8, "tps": 28.0, "latency": 1.2},
                      "mistral:7b":  {"quality": 2.2, "tps": 18.0, "latency": 2.0}},
    }
    return defaults


# ── Routing logic ─────────────────────────────────────────────────────────────

def pick_model(
    category: str,
    routing_table: dict,
    prefer_speed: bool = False,
) -> RoutingDecision:
    """
    Pick the best model for a given category.

    Default: maximise quality score, tiebreak on TPS.
    prefer_speed=True: maximise TPS, tiebreak on quality.
    """
    candidates = routing_table.get(category, {})

    # Fallback to factual if category not in table
    if not candidates:
        candidates = routing_table.get("factual", {})

    if not candidates:
        return RoutingDecision(
            chosen_model="llama3.2:3b",
            category=category,
            reason="No benchmark data — using default model",
            quality_score=0.0,
            tps=0.0,
            alternatives=[],
        )

    # Score each model
    ranked = []
    for model, stats in candidates.items():
        ranked.append({
            "model":   model,
            "quality": stats["quality"],
            "tps":     stats["tps"],
            "latency": stats["latency"],
        })

    if prefer_speed:
        ranked.sort(key=lambda x: (x["tps"], x["quality"]), reverse=True)
        reason_template = "Fastest model for {cat} ({tps:.1f} tok/s)"
    else:
        ranked.sort(key=lambda x: (x["quality"], x["tps"]), reverse=True)
        reason_template = "Highest quality for {cat} (score {q:.2f}/3)"

    best = ranked[0]
    label = MODELS.get(best["model"], {}).get("label", best["model"])

    return RoutingDecision(
        chosen_model=best["model"],
        category=category,
        reason=reason_template.format(
            cat=category, tps=best["tps"], q=best["quality"]
        ) + f" — {label}",
        quality_score=best["quality"],
        tps=best["tps"],
        alternatives=ranked[1:],
    )


# ── Router ────────────────────────────────────────────────────────────────────

class Router:
    """
    Main router object. Instantiate once, call .route() for each question.
    Loads benchmark data on init so routing is instant per-query.
    """

    def __init__(self, prefer_speed: bool = False):
        self.prefer_speed  = prefer_speed
        self.routing_table = load_benchmark_scores()
        self.client        = ollama.Client()
        log.info(
            "Router initialised | categories: %s | prefer_speed: %s",
            list(self.routing_table.keys()), prefer_speed,
        )

    def route(self, question: str) -> RoutedResponse:
        """
        Full pipeline: classify → route → call model → return response.
        """
        # Step 1 — classify
        classification = classify(question)

        # Step 2 — pick model
        decision = pick_model(
            classification.category,
            self.routing_table,
            prefer_speed=self.prefer_speed,
        )

        # Step 3 — call chosen model
        t0 = time.perf_counter()
        try:
            resp = self.client.chat(
                model=decision.chosen_model,
                messages=[{"role": "user", "content": question}],
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

    def explain(self, question: str) -> dict:
        """
        Returns routing explanation without actually calling the model.
        Useful for debugging routing decisions.
        """
        classification = classify(question)
        decision       = pick_model(
            classification.category,
            self.routing_table,
            prefer_speed=self.prefer_speed,
        )
        return {
            "question":   question,
            "category":   classification.category,
            "confidence": classification.confidence,
            "chosen":     decision.chosen_model,
            "reason":     decision.reason,
            "alternatives": decision.alternatives,
        }
