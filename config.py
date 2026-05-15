"""
config.py — Shared configuration, constants, and utility helpers
"""

import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─── Directories ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Models under test ──────────────────────────────────────────────────────
MODELS = {
    "llama3.2:3b":  {"label": "Llama 3.2 3B",  "params_b": 3.2},
    "phi4-mini":    {"label": "Phi-4 Mini",     "params_b": 3.8},
    "mistral:7b":   {"label": "Mistral 7B",     "params_b": 7.0},
}

# Default model for Phase 1 & 2 development work
DEFAULT_MODEL = "llama3.2:3b"

# ─── Temperature ladder (Phase 2) ───────────────────────────────────────────
TEMPERATURES = [0.0, 0.3, 0.7, 1.0, 1.4]

# ─── Ollama host ─────────────────────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("local_ai")


# ─── Core measurement dataclass ──────────────────────────────────────────────
@dataclass
class LatencyRecord:
    model:               str
    prompt_id:           str
    prompt_text:         str
    temperature:         float
    time_to_first_token: float          # seconds
    total_latency:       float          # seconds (wall-clock, generation only)
    prompt_tokens:       int
    completion_tokens:   int
    tokens_per_second:   float
    response_text:       str
    error:               Optional[str] = None
    phase:               str = "phase1"
    extra:               dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def save_jsonl(records: list, path: Path) -> None:
    """Append a list of dicts to a .jsonl file."""
    with open(path, "a") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    log.info("Saved %d records → %s", len(records), path)


def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a .jsonl file."""
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def ms(seconds: float) -> str:
    """Format seconds → human-readable ms string."""
    return f"{seconds * 1000:.1f} ms"
