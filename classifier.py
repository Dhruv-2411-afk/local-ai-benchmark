"""
classifier.py
─────────────
Classifies an incoming question into one of five categories:
  factual | reasoning | code | creative | edge

Uses the local Llama model itself to classify — no external API needed.
Returns a confidence score alongside the category.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

import ollama

from config import DEFAULT_MODEL, log

# ── Prompt ────────────────────────────────────────────────────────────────────

CLASSIFIER_PROMPT = """You are a question classifier. Classify the given question into exactly one category.

Categories:
- factual    : questions with a single correct answer (history, science, definitions, facts)
- reasoning  : math, logic puzzles, step-by-step problems, probability
- code       : programming, debugging, algorithms, SQL, regex, shell scripts
- creative   : writing, brainstorming, analogies, stories, names, descriptions
- edge       : trick questions, very short answers, translations, sequences, yes/no

Respond ONLY with valid JSON, no extra text:
{
  "category": "<one of: factual, reasoning, code, creative, edge>",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining why>"
}

Question to classify: """


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    category:   str
    confidence: float
    reason:     str
    raw:        str         # raw model output for debugging

    def is_confident(self, threshold: float = 0.6) -> bool:
        return self.confidence >= threshold


# ── Keyword fallback (instant, no model call) ─────────────────────────────────

KEYWORD_RULES: list[tuple[list[str], str]] = [
    (["write", "function", "code", "implement", "algorithm", "sql",
      "query", "debug", "class", "script", "regex", "bash", "python",
      "javascript", "def ", "loop", "array"], "code"),
    (["calculate", "solve", "probability", "how many", "if a", "if an",
      "math", "equation", "proof", "logic", "steps", "derive"], "reasoning"),
    (["who is", "what is the capital", "when did", "how old",
      "what year", "name the", "define", "what does", "stand for",
      "how many bones", "speed of"], "factual"),
    (["write a story", "poem", "haiku", "describe", "suggest",
      "invent", "create a name", "analogy", "pitch", "proverb",
      "opening line"], "creative"),
    (["yes or no", "only with", "translate", "next three",
      "rhymes with", "nato", "sequence", "continue this"], "edge"),
]


def keyword_classify(question: str) -> Optional[str]:
    """Fast keyword-based pre-classifier. Returns None if unsure."""
    q = question.lower()
    for keywords, category in KEYWORD_RULES:
        if any(kw in q for kw in keywords):
            return category
    return None


# ── Main classifier ───────────────────────────────────────────────────────────

def classify(
    question: str,
    model: str = DEFAULT_MODEL,
    use_keyword_shortcut: bool = True,
) -> ClassificationResult:
    """
    Classify a question into one of five categories.

    Strategy:
      1. Try keyword rules first (instant, ~0ms)
      2. If confident keyword match → return immediately
      3. Otherwise → ask the local model (adds ~500ms)
    """

    # ── Step 1: keyword shortcut ──────────────────────────────────────────────
    if use_keyword_shortcut:
        kw_cat = keyword_classify(question)
        if kw_cat:
            log.debug("Keyword classified '%s' → %s", question[:40], kw_cat)
            return ClassificationResult(
                category=kw_cat,
                confidence=0.80,
                reason="Matched keyword pattern",
                raw="[keyword]",
            )

    # ── Step 2: model-based classification ───────────────────────────────────
    client = ollama.Client()
    prompt = CLASSIFIER_PROMPT + f'"{question}"'

    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "seed": 42},
        )
        raw = resp["message"]["content"]

        # Extract JSON
        text = re.sub(r"```(?:json)?", "", raw).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return ClassificationResult(
                category=data.get("category", "factual"),
                confidence=float(data.get("confidence", 0.7)),
                reason=data.get("reason", ""),
                raw=raw,
            )

    except Exception as exc:
        log.warning("Classifier model call failed: %s", exc)

    # ── Step 3: fallback default ──────────────────────────────────────────────
    return ClassificationResult(
        category="factual",
        confidence=0.4,
        reason="Classification failed — defaulting to factual",
        raw="[error]",
    )


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "What is the capital of France?",
        "Write a Python function to reverse a linked list",
        "If I flip a coin 4 times what is the probability of all heads?",
        "Write a haiku about debugging code",
        "Respond only with YES or NO: Is the sun a star?",
    ]
    print("\nClassifier Test\n" + "─" * 40)
    for q in tests:
        result = classify(q)
        print(f"Q: {q}")
        print(f"   → {result.category} (confidence: {result.confidence:.0%}) | {result.reason}\n")
