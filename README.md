# Local AI Benchmark & Smart Router

A three-phase benchmarking system for small language models (3-7B parameters) running locally via [Ollama](https://ollama.ai), with a smart router that uses the benchmark results to automatically pick the best model for each query.

> **No internet. No API keys. No cost. Runs entirely on your machine.**

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
- **Mistral 7B** is the slowest on this hardware but has near-zero memory overhead
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

## Smart Router

The router uses Phase 3 quality scores to automatically select the best model per question type.

```
You ask a question
       |
Classifier (keyword rules + local model fallback)
       |
Category: factual | reasoning | code | creative | edge
       |
Routing table (built from your benchmark data)
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

### Example session

```
You: Write a binary search in Python
  -> classified as code  -> routed to Mistral 7B  (14.2s | 12 tok/s)

You: What year did the Berlin Wall fall?
  -> classified as factual  -> routed to Llama 3.2 3B  (1.1s | 44 tok/s)

You: If I flip a coin 3 times what is the probability of all heads?
  -> classified as reasoning  -> routed to Phi-4 Mini  (8.3s | 28 tok/s)
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

# 5. Start the assistant
python assistant.py
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Ollama | Local model serving |
| Pydantic v2 | Response schema validation |
| psutil | Memory profiling |
| Rich | Terminal UI |
| Python 3.10+ | Everything else |

---

## Models Tested

| Model | Parameters | Best for |
|-------|----------:|---------|
| Llama 3.2 3B | 3.2B | Speed - fastest response, lowest latency |
| Phi-4 Mini | 3.8B | Quality - most detailed, highest accuracy |
| Mistral 7B | 7.0B | Balanced - strong on code and reasoning |
