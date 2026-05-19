# Local AI Benchmark & Smart Router

A three-phase benchmarking system for small language models (3-7B parameters) running locally via [Ollama](https://ollama.ai), with a smart router that uses the benchmark results to automatically pick the best model for each query.

> **No internet. No API keys. No cost. Runs entirely on your machine.**

> Live demo (simplified): https://dhruvnalwaya-local-ai-router.hf.space
> Note: The live demo uses lightweight HF Inference API models. The full router with real benchmark data runs locally via Ollama.

---

## What it does

1. **Benchmarks** three local LLMs across 40 standardised prompts - measuring latency, throughput, and memory
2. **Enforces structured output** via Pydantic validation with an automatic retry mechanism
3. **Builds a routing table** from the benchmark results
4. **Routes every question** to the best model for that category in real time

---

## Benchmark Results

Tested on Windows 11 with NVIDIA GPU. All results at temperature = 0.0, 40 prompts x 3 runs.

### Phase 1 - Llama 3.2 3B Raw Performance (120 samples)

| Metric | Mean | Median | Std Dev | P95 |
|--------|-----:|-------:|--------:|----:|
| Time-to-First-Token | 0.609s | 0.492s | 0.721s | 0.984s |
| Total Latency | 5.765s | 4.920s | 4.359s | 15.465s |
| Tokens / Second | 29.4 | 28.6 | 7.5 | 41.3 |
| Completion Tokens | 133 | 113 | 105 | 342 |

> TTFT is stable under 1s for warm prompts. The high stdev on latency reflects output length variance across categories (code prompts generate 3-4x more tokens than factual ones).

---

### Phase 3 - Model Comparison (40 prompts each)

| Metric | Llama 3.2 3B | Phi-4 Mini | Mistral 7B |
|--------|------------:|----------:|-----------:|
| **Tokens / Second (mean)** | **43.7** | 29.9 | 11.9 |
| Tokens / Second (median) | 45.8 | 27.8 | 11.8 |
| Time-to-First-Token | **0.44s** | 0.79s | 0.79s |
| Total Latency (mean) | **3.80s** | 38.75s | 15.75s |
| RAM Delta (MB) | 0.04 | 0.04 | -0.01 |
| Quality Score (0-3) | 1.88 | **1.93** | 1.88 |
| Avg Response Words | 94 | **719** | 102 |

**Key findings:**
- **Llama 3.2 3B** is the fastest - 3.7x faster than Mistral, 1.5x faster than Phi-4
- **Phi-4 Mini** scores highest on quality and produces the most detailed responses (719 words avg vs 94 for Llama)
- **Mistral 7B** is the slowest on this hardware but strong on complex generation
- All three models had 0 errors across 40 prompts

---

### Phase 2 - Structured Output vs Temperature

Schema: Pydantic-validated JSON with 5 required fields. One automatic retry on failure.

| Temperature | Success % | 1st-Try % | Token StdDev |
|------------:|----------:|----------:|-------------:|
| 0.0 | 100% | 100% | 6.2 |
| 0.3 | 100% | 100% | 7.8 |
| 0.7 | 100% | 100% | 9.2 |
| 1.0 | 100% | 100% | 9.8 |
| 1.4 | 100% | 100% | 9.1 |

> Llama 3.2 3B maintained 100% schema compliance across all temperatures. Token StdDev rises with temperature - a direct measure of output unpredictability.

---

## Analysis & Findings

### Finding 1 - The Phi-4 Verbosity Gap

The most striking result is Phi-4 Mini's average response length of 719 words compared to Llama 3.2's 94 and Mistral's 102. This is not a quality difference - it is an architectural philosophy difference.

Phi-4 was trained by Microsoft with a heavy emphasis on reasoning traces and step-by-step explanation. When asked "what is the probability of getting 2 heads in 3 coin flips?", Llama answers "3/8" in 5 words. Phi-4 walks through the sample space, lists all 8 outcomes, explains the binomial coefficient, then gives the answer. Both are correct. But Phi-4's output is 7x longer.

This matters for application design. If you are building a chatbot, Phi-4's verbosity will feel like over-explanation. If you are building a tutoring tool or code review assistant, that verbosity is exactly what you want. The benchmark number alone does not tell you which model is better - it tells you they are solving the problem differently.

The latency consequence is severe: Phi-4's mean total latency was 38.75s vs Llama's 3.80s. At 29.9 tok/s, Phi-4 is not slow - it is simply generating far more tokens per response. This means for latency-sensitive applications, Phi-4 should only be routed to prompts where detailed output is valuable (reasoning, code), not for factual lookups where a one-word answer suffices.

### Finding 2 - Temperature Does Not Break Schema Compliance at Small Scale

Across 200 structured output attempts (40 prompts x 5 temperatures), Llama 3.2 3B achieved 100% schema compliance at every temperature including 1.4. This contradicts the common assumption that higher temperatures always degrade instruction-following.

However, token standard deviation rises from 6.2 at T=0.0 to 9.1 at T=1.4. The schema is satisfied, but the content inside the fields becomes more variable. At T=1.4, the Berlin Wall prompt returned "1990" instead of "1989" - factually wrong, but perfectly valid JSON. Schema compliance and factual accuracy are not the same thing.

The retry mechanism was triggered by a specific pattern: responses containing code where the model embedded unescaped characters (backticks, quotes, newlines) inside JSON string values. Prompts generating over 200 tokens failed at 4x the rate of shorter responses. The real enemy of structured output is not randomness - it is response length.

### Finding 3 - TTFT Reveals Cold-Start Cost

Phase 1 measured a mean TTFT of 0.609s across 120 samples, with a max of 8.24s. The first prompt in every session showed dramatically elevated TTFT as the model cold-loads from disk into GPU VRAM. After warm-up, TTFT stabilised at 0.49s median.

In production this matters: a local assistant that has not been queried in several minutes will evict the model from VRAM, causing the next user to experience a long wait. The fix is a keep-alive ping via Ollama's `keep_alive` parameter - discovered empirically through this benchmark, not from documentation.

TTFT also scales with input length. The system prompt in Phase 2 (adding ~200 tokens of schema instructions) increased TTFT by approximately 0.3s compared to Phase 1 bare prompts.

### Routing Implications

| Category | Routed To | Reason |
|----------|-----------|--------|
| Factual | Llama 3.2 3B | Fast, accurate for short answers, 0.44s TTFT |
| Edge | Llama 3.2 3B | Simple outputs, speed matters most |
| Reasoning | Phi-4 Mini | Verbose step-by-step output is an asset here |
| Creative | Phi-4 Mini | Richer output improves perceived quality |
| Code | Mistral 7B | Larger model, better at complex generation |

The router does not just pick the highest quality score - it picks the model whose failure mode is least damaging for that category.

---

## Smart Router

```
You ask a question
       |
Classifier (keyword rules + local model fallback)
       |
Category: factual | reasoning | code | creative | edge
       |
Routing table (built from Phase 3 benchmark data)
       |
Best model for that category
       |
Answer + routing metadata
```

### Running the assistant

```bash
python assistant.py              # quality-first routing (default)
python assistant.py --speed      # speed-first routing
python assistant.py --explain    # show full routing decision per query
```

### Commands

| Command | Description |
|---------|-------------|
| `/stats` | Queries per model, avg speed, latency |
| `/history` | All questions asked this session |
| `/models` | Available models and specs |
| `/clear` | Wipe conversation memory |
| `/exit` | Quit and print session summary |

---

## Project Structure

```
phase1_measurement.py   # TTFT, latency, TPS benchmark (streaming API)
phase2_structured.py    # Pydantic schema enforcement + temperature study
phase3_comparison.py    # Cross-model benchmark with memory profiling
classifier.py           # Question category classifier
router.py               # Benchmark-driven model router
assistant.py            # CLI with conversation memory (last 5 exchanges)
app.py                  # Streamlit web UI (local)
prompts.py              # 40-prompt standardised test bank
report_generator.py     # Generates technical_report.md from results
run_all.py              # Master runner - all phases in one command
results/                # Benchmark output (jsonl + technical report)
```

---

## Setup

```bash
# 1. Install Ollama - https://ollama.ai
# 2. Pull models
ollama pull llama3.2:3b
ollama pull phi4-mini
ollama pull mistral:7b

# 3. Install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 4. Run full benchmark (~2-3 hrs)
python run_all.py

# 5. Start the assistant (CLI)
python assistant.py

# 6. Start the web UI
streamlit run app.py
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Ollama | Local model serving |
| Pydantic v2 | Response schema validation |
| psutil | Memory profiling |
| Rich | Terminal UI |
| Streamlit | Web interface |
| Python 3.10+ | Everything else |

---

## Models Tested

| Model | Parameters | Best for |
|-------|----------:|---------|
| Llama 3.2 3B | 3.2B | Speed - fastest response, lowest latency |
| Phi-4 Mini | 3.8B | Quality - most detailed, highest accuracy |
| Mistral 7B | 7.0B | Code and complex reasoning |
