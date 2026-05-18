"""
phase4_classifier.py
────────────────────
Classifies any user question into one of five categories:
  factual | reasoning | code | creative | edge

Uses the local model itself to classify — no external APIs, no hardcoded
keyword lists. The classifier is a zero-shot prompt with strict JSON output.

Why use the model to classify?
  Keyword matching ("does it contain 'write a function'?") breaks on
  paraphrasing. The model understands intent, not just surface words.

Usage:
    from phase4_classifier import classify
    result = classify("What is the speed of light?")
    # → ClassificationResult(category='factual', confidence=0.95, ...)

    # Or run standalone:
    python phase4_classifier.py
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

import ollama

from config import DEFAULT_MODEL, log

# ─── Classifier prompt ────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = """You are a question classifier. Given a user question, 
classify it into exactly one of these five categories:

  factual   — questions with a single correct answer (history, science, math facts)
  reasoning — questions requiring logic, multi-step thinking, or analysis  
  code      — questions asking to write, debug, or explain code or SQL
  creative  — questions asking for writing, ideas, analogies, or creative output
  edge      — short commands, yes/no questions, translations, or unusual requests

Respond ONLY with valid JSON, no other text:
{
  "category": "<one of: factual, reasoning, code, creative, edge>",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining your choice>"
}"""


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    category:   str
    confidence: float
    reason:     str
    latency_s:  float
    raw:        str
    error:      Optional[str] = None


# ─── Core classifier ──────────────────────────────────────────────────────────

def classify(
    question: str,
    model: str = DEFAULT_MODEL,
) -> ClassificationResult:
    """
    Classify a question using the local model.
    Falls back to 'reasoning' if classification fails.
    """
    client = ollama.Client()
    t0     = time.perf_counter()

    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system",  "content": CLASSIFIER_SYSTEM},
                {"role": "user",    "content": f"Classify this question: {question}"},
            ],
            options={"temperature": 0.0, "seed": 42},
        )
        raw     = resp["message"]["content"]
        latency = time.perf_counter() - t0

        # Extract JSON
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        match   = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            data = json.loads(cleaned)

        category   = data.get("category", "reasoning")
        confidence = float(data.get("confidence", 0.5))
        reason     = data.get("reason", "")

        # Validate category
        valid = {"factual", "reasoning", "code", "creative", "edge"}
        if category not in valid:
            category = "reasoning"

        return ClassificationResult(
            category=category,
            confidence=confidence,
            reason=reason,
            latency_s=round(latency, 3),
            raw=raw,
        )

    except Exception as exc:
        latency = time.perf_counter() - t0
        log.warning("Classifier error: %s — defaulting to 'reasoning'", exc)
        return ClassificationResult(
            category="reasoning",
            confidence=0.0,
            reason="Classification failed, using default",
            latency_s=round(latency, 3),
            raw="",
            error=str(exc),
        )


# ─── Standalone test ──────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    "What is the capital of Japan?",
    "Write a Python function to reverse a linked list",
    "If all cats are animals and some animals are dogs, are some cats dogs?",
    "Write a haiku about debugging at 3am",
    "Is the sky blue? Answer only YES or NO",
    "What is 17 multiplied by 23?",
    "Explain recursion using a real-world analogy",
    "Write a SQL query to find duplicate emails in a users table",
]

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.rule("[bold]Phase 4 — Question Classifier Test[/bold]")

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Question",    width=45)
    table.add_column("Category",    style="bold cyan")
    table.add_column("Confidence",  justify="right")
    table.add_column("Latency",     justify="right")
    table.add_column("Reason",      width=35)

    for q in TEST_QUESTIONS:
        result = classify(q)
        table.add_row(
            q[:44],
            result.category,
            f"{result.confidence:.0%}",
            f"{result.latency_s:.2f}s",
            result.reason[:34],
        )

    console.print(table)
