# Local AI Assistant — Benchmark Suite

Three-phase benchmark for small language models (3–7 B) running locally via Ollama.

---

## Prerequisites

```bash
# 1. Install Ollama  →  https://ollama.ai
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull the three models (~8 GB total)
ollama pull llama3.2:3b
ollama pull phi4-mini
ollama pull mistral:7b
```

---

## Project Structure

```
local_ai_assistant/
├── config.py               # Shared constants, LatencyRecord dataclass, helpers
├── prompts.py              # 40-prompt bank across 5 categories
│
├── phase1_measurement.py   # TTFT / latency / TPS benchmark (streaming)
├── phase2_structured.py    # JSON schema enforcement, Pydantic, retry, T study
├── phase3_comparison.py    # Cross-model comparison + memory profiling
│
├── report_generator.py     # Reads results/, writes technical_report.md
├── run_all.py              # Master runner — runs all phases in sequence
│
├── requirements.txt
├── results/                # Created automatically
│   ├── phase1_results.jsonl
│   ├── phase1_summary.json
│   ├── phase2_results.jsonl
│   ├── phase3_llama3.2_3b.jsonl
│   ├── phase3_phi4-mini.jsonl
│   ├── phase3_mistral_7b.jsonl
│   ├── phase3_aggregates.json
│   └── technical_report.md
```

---

## Running

### Full benchmark (recommended — takes 2–4 hours)
```bash
python run_all.py
```

### Quick smoke-test (5 prompts, llama only, ~10 minutes)
```bash
python run_all.py --quick
```

### Individual phases
```bash
# Phase 1 only
python phase1_measurement.py --model llama3.2:3b --runs 3

# Phase 2 only (all temperatures)
python phase2_structured.py --model llama3.2:3b --temps 0.0 0.3 0.7 1.0 1.4

# Phase 3 only (all three models)
python phase3_comparison.py --models llama3.2:3b phi4-mini mistral:7b

# Generate report from existing results
python report_generator.py --from-results
```

---

## Phase Overview

### Phase 1 — Raw Performance Measurement
Runs every prompt `--runs` times using the **streaming API** (mandatory for TTFT).
Captures:
- **TTFT** — time_to_first_token: latency from dispatch to first byte
- **Total latency** — wall-clock time until generation completes
- **TPS** — completion_tokens ÷ generation_time

Results: per-prompt records + mean/median/std/P95 summary per model.

### Phase 2 — Structured Output & Temperature Study
Enforces a **strict JSON schema** via a system prompt, validates with **Pydantic**,
and retries once on failure before logging a graceful error.

Temperatures tested: `[0.0, 0.3, 0.7, 1.0, 1.4]`

Documents:
- Success rate and first-try rate per temperature
- Token count variance (std dev) as a proxy for output unpredictability
- Retry recovery rate
- Confidence self-rating distribution

### Phase 3 — Model Comparison Study
Benchmarks all three models on the same 40-prompt bank:
- **Performance:** TPS, TTFT, total latency
- **Memory:** process RSS delta per call (via `psutil`)
- **Quality:** automated 0–3 score (length + keyword + schema checks)

Produces a leaderboard and per-category quality breakdown.

---

## Tips

- **GPU acceleration**: Ollama auto-detects CUDA / Metal. Check `ollama ps` to
  see which layers are on GPU.
- **Memory**: Mistral 7B requires ~6 GB RAM (CPU) or VRAM (GPU). Ensure you
  have headroom before running Phase 3.
- **Reproducibility**: Temperature 0.0 + `seed=42` are set in all Phase 1 & 3
  calls for maximum reproducibility.
- **Adding models**: Add entries to `MODELS` in `config.py` and pass them via
  `--models` to `phase3_comparison.py`.

---

## Report

After a full run, the Markdown technical report is at:
```
results/technical_report.md
```
It includes actual numbers pulled from all `.jsonl` result files with analysis,
winner declarations, and recommendations.
