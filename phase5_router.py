"""
phase5_router.py
────────────────
The routing engine — the heart of Option C.

How it works:
  1. Reads Phase 3 benchmark results (phase3_aggregates.json +
     per-model .jsonl files) to build a routing table
  2. For each category, ranks models by a weighted score:
       score = (quality_weight × quality) + (speed_weight × normalised_tps)
  3. When a question arrives, the classifier's category is looked up in
     the routing table and the top-ranked model is selected
  4. The question is sent to that model and the response is returned
     alongside full routing metadata (why this model, what alternatives were)

The routing table is built once at startup and cached — zero overhead
per query after initialisation.

Usage:
    from phase5_router import Router
    router = Router()
    result = router.route("Write a binary search in Python")
    print(result.answer)
    print(result.routing_metadata)
"""

import json
import time
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import ollama

from config import MODELS, RESULTS_DIR, log
from phase4_classifier import classify, ClassificationResult

# ─── Routing weights (tunable) ────────────────────────────────────────────────
# Adjust these to bias the router toward quality vs speed
QUALITY_WEIGHT = 0.65
SPEED_WEIGHT   = 0.35

# Per-category overrides — some tasks always prefer speed or quality
CATEGORY_OVERRIDES: dict[str, dict] = {
    "code":      {"quality_weight": 0.80, "speed_weight": 0.20},
    "reasoning": {"quality_weight": 0.80, "speed_weight": 0.20},
    "factual":   {"quality_weight": 0.50, "speed_weight": 0.50},
    "creative":  {"quality_weight": 0.70, "speed_weight": 0.30},
    "edge":      {"quality_weight": 0.30, "speed_weight": 0.70},
}


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ModelScore:
    model:          str
    label:          str
    quality_mean:   float
    tps_mean:       float
    latency_mean:   float
    composite_score: float


@dataclass
class RoutingDecision:
    category:        str
    selected_model:  str
    selected_label:  str
    reason:          str
    alternatives:    list[dict]      # other models considered
    scores:          list[dict]      # full score breakdown


@dataclass
class RouterResult:
    question:          str
    classification:    ClassificationResult
    routing:           RoutingDecision
    answer:            str
    total_latency_s:   float
    inference_latency_s: float
    completion_tokens: int
    tokens_per_second: float
    error:             Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "question":            self.question,
            "category":            self.classification.category,
            "selected_model":      self.routing.selected_model,
            "reason":              self.routing.reason,
            "answer":              self.answer,
            "total_latency_s":     self.total_latency_s,
            "inference_latency_s": self.inference_latency_s,
            "tokens_per_second":   self.tokens_per_second,
            "completion_tokens":   self.completion_tokens,
            "error":               self.error,
        }

    def pretty(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  Question  : {self.question}",
            f"  Category  : {self.classification.category} "
            f"(confidence {self.classification.confidence:.0%})",
            f"  Routed to : {self.routing.selected_label}",
            f"  Why       : {self.routing.reason}",
            f"{'─'*60}",
            f"\n{self.answer}\n",
            f"{'─'*60}",
            f"  Latency   : {self.total_latency_s:.2f}s total  "
            f"({self.inference_latency_s:.2f}s inference)",
            f"  Speed     : {self.tokens_per_second:.1f} tok/s  "
            f"| {self.completion_tokens} tokens",
            f"{'─'*60}\n",
        ]
        return "\n".join(lines)


# ─── Routing table builder ────────────────────────────────────────────────────

class RoutingTable:
    """
    Builds and stores per-category model rankings from Phase 3 results.
    Falls back to sensible defaults if results don't exist yet.
    """

    def __init__(self) -> None:
        self.table: dict[str, list[ModelScore]] = {}
        self._build()

    def _build(self) -> None:
        agg_path = RESULTS_DIR / "phase3_aggregates.json"
        categories = ["factual", "reasoning", "code", "creative", "edge"]

        if not agg_path.exists():
            log.warning("phase3_aggregates.json not found — using default routing")
            self._build_defaults()
            return

        aggs = json.loads(agg_path.read_text(encoding="utf-8"))

        # Collect per-category quality from individual jsonl files
        cat_quality: dict[str, dict[str, float]] = {c: {} for c in categories}

        for model_key in MODELS:
            safe = model_key.replace(":", "_").replace("/", "_")
            path = RESULTS_DIR / f"phase3_{safe}.jsonl"
            if not path.exists():
                continue
            records = [
                json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            for cat in categories:
                cat_recs = [r for r in records if r.get("category") == cat]
                if cat_recs:
                    cat_quality[cat][model_key] = statistics.mean(
                        r.get("quality_score", 0) for r in cat_recs
                    )

        # Get all TPS values for normalisation
        all_tps = [
            aggs[m].get("tps_mean", 0)
            for m in aggs
            if aggs[m].get("tps_mean", 0) > 0
        ]
        max_tps = max(all_tps) if all_tps else 1.0

        for cat in categories:
            weights = CATEGORY_OVERRIDES.get(
                cat, {"quality_weight": QUALITY_WEIGHT, "speed_weight": SPEED_WEIGHT}
            )
            qw = weights["quality_weight"]
            sw = weights["speed_weight"]

            scores = []
            for model_key, info in MODELS.items():
                a = aggs.get(model_key, {})
                if not a:
                    continue
                q   = cat_quality[cat].get(model_key, a.get("quality_mean", 0))
                tps = a.get("tps_mean", 0)
                norm_tps  = tps / max_tps          # 0-1
                norm_qual = q   / 3.0              # 0-1 (max quality is 3)
                composite = qw * norm_qual + sw * norm_tps

                scores.append(ModelScore(
                    model          = model_key,
                    label          = info["label"],
                    quality_mean   = round(q,   3),
                    tps_mean       = round(tps, 2),
                    latency_mean   = round(a.get("latency_mean_s", 0), 3),
                    composite_score= round(composite, 4),
                ))

            # Sort best first
            scores.sort(key=lambda s: s.composite_score, reverse=True)
            self.table[cat] = scores

        log.info("Routing table built from Phase 3 results")
        self._log_table()

    def _build_defaults(self) -> None:
        """Fallback when no Phase 3 data exists — prefer smaller/faster models."""
        categories = ["factual", "reasoning", "code", "creative", "edge"]
        default_order = list(MODELS.keys())
        for cat in categories:
            self.table[cat] = [
                ModelScore(
                    model=m, label=MODELS[m]["label"],
                    quality_mean=0, tps_mean=0,
                    latency_mean=0, composite_score=0,
                )
                for m in default_order
            ]

    def _log_table(self) -> None:
        for cat, scores in self.table.items():
            if scores:
                log.info(
                    "%-10s → best: %-14s (score %.3f)",
                    cat, scores[0].label, scores[0].composite_score,
                )

    def best_model(self, category: str) -> ModelScore:
        scores = self.table.get(category, [])
        if scores:
            return scores[0]
        # Ultimate fallback
        first_model = next(iter(MODELS))
        return ModelScore(
            model=first_model,
            label=MODELS[first_model]["label"],
            quality_mean=0, tps_mean=0,
            latency_mean=0, composite_score=0,
        )

    def routing_decision(self, category: str) -> RoutingDecision:
        scores = self.table.get(category, [])
        best   = self.best_model(category)
        weights = CATEGORY_OVERRIDES.get(
            category,
            {"quality_weight": QUALITY_WEIGHT, "speed_weight": SPEED_WEIGHT}
        )

        reason = (
            f"{best.label} scored highest for '{category}' tasks "
            f"(quality weight={weights['quality_weight']:.0%}, "
            f"speed weight={weights['speed_weight']:.0%})"
        )

        return RoutingDecision(
            category       = category,
            selected_model = best.model,
            selected_label = best.label,
            reason         = reason,
            alternatives   = [
                {"model": s.model, "label": s.label,
                 "score": s.composite_score}
                for s in scores[1:]
            ],
            scores         = [
                {"model": s.model, "label": s.label,
                 "quality": s.quality_mean, "tps": s.tps_mean,
                 "composite": s.composite_score}
                for s in scores
            ],
        )


# ─── Router ───────────────────────────────────────────────────────────────────

class Router:
    """
    Main router class. Initialise once, call route() for every question.

    Example:
        router = Router()
        result = router.route("Explain quicksort")
        print(result.pretty())
    """

    def __init__(self, classifier_model: str = None) -> None:
        self.client           = ollama.Client()
        self.classifier_model = classifier_model or list(MODELS.keys())[0]
        self.routing_table    = RoutingTable()
        log.info("Router ready — classifier: %s", self.classifier_model)

    def route(self, question: str) -> RouterResult:
        t_total_start = time.perf_counter()

        # ── Step 1: Classify ──────────────────────────────────────────────────
        classification = classify(question, model=self.classifier_model)

        # ── Step 2: Decide which model ────────────────────────────────────────
        decision = self.routing_table.routing_decision(classification.category)

        # ── Step 3: Inference ─────────────────────────────────────────────────
        t_infer_start = time.perf_counter()
        answer        = ""
        tokens        = 0
        tps           = 0.0
        error         = None

        try:
            resp = self.client.chat(
                model   = decision.selected_model,
                messages= [{"role": "user", "content": question}],
                options = {"temperature": 0.7, "seed": 42},
            )
            answer = resp["message"]["content"]
            tokens = resp.get("eval_count", 0)
            infer_time = time.perf_counter() - t_infer_start
            tps = tokens / infer_time if infer_time > 0 else 0.0

        except Exception as exc:
            error      = str(exc)
            infer_time = time.perf_counter() - t_infer_start
            log.error("Inference error: %s", exc)

        total_time = time.perf_counter() - t_total_start

        return RouterResult(
            question            = question,
            classification      = classification,
            routing             = decision,
            answer              = answer,
            total_latency_s     = round(total_time,  3),
            inference_latency_s = round(infer_time,  3),
            completion_tokens   = tokens,
            tokens_per_second   = round(tps, 2),
            error               = error,
        )
