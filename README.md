# ConjLean / REFUTE

**Two systems. One goal: automated mathematics.**

> **ICML AI4Research 2026 submission** — *REFUTE: Property-Based Testing for Mathematics via Multi-Agent Counterexample Search*

---

## Systems at a Glance

| System | What it does |
|--------|-------------|
| **ConjLean** | Generate → SymPy-filter → Lean 4 formalize → 4-layer proof search |
| **REFUTE** | Load conjecture → R-Agent searches for counterexamples → V-Agent verifies → C-Agent refines → S-Agent orchestrates |

ConjLean tries to **prove** conjectures. REFUTE tries to **refute** them — finding the counterexamples that show where a conjecture breaks down. Both pipelines share the same LLM backend, config, and schemas.

---

## REFUTE Architecture

```
                 ┌──────────────────────────────────────────┐
                 │              REFUTE Loop                  │
                 │                                           │
  Conjecture ──► │  S-Agent (UCB1 meta-controller)          │
                 │      │                                    │
                 │      ▼                                    │
                 │  R-Agent  ──── 4 strategies ────►  CE?   │
                 │  (Refuter)   BOUNDARY                │    │
                 │              RANDOM_STRUCTURED        │    │
                 │              ANALOGICAL               │    │
                 │              SYMBOLIC_PERTURBATION    │    │
                 │                                       │    │
                 │      V-Agent (SymPy verifier) ◄───────┘    │
                 │      │  confirmed?                         │
                 │      ▼  yes                                │
                 │  C-Agent (Conjecturer/refiner)             │
                 │      │  tightens hypothesis               │
                 │      └──► refined conjecture ──► repeat   │
                 │                                           │
                 │  Terminal states: REFUTED · REFINED       │
                 │                   SURVIVED · BUDGET_EXHAUSTED
                 └──────────────────────────────────────────┘
```

**R-Agent strategies**

| Strategy | Description | LLM? |
|----------|-------------|------|
| `BOUNDARY` | Deterministic sweep of edge-case integers / extremes | No |
| `RANDOM_STRUCTURED` | Primes, squares, Fibonacci, factorials | No |
| `ANALOGICAL` | LLM proposes candidates based on past counterexamples | Yes |
| `SYMBOLIC_PERTURBATION` | LLM identifies critical parameters and perturbs | Yes |

**S-Agent (Strategist) selection cascade:** BOUNDARY first → domain win-rate check → LLM if stats sparse → UCB1 exploitation/exploration balance.

---

## ConjLean Architecture

```
LLM  ──►  SymPy filter  ──►  Lean 4 autoformalization  ──►  4-layer proof search
Generate   Pre-check          NL → Lean4                     Layer 0: omega/norm_num/decide
           (CPU, ~5 s)        (error-repair loop)            Layer 1: induction combos
                                                             Layer 2: exact?/apply? search
                                                             Layer 3: LLM + Lean REPL
```

---

## REFUTE Benchmark

The 3-tier benchmark is built from `scripts/build_benchmark.py` and lives in `data/benchmark/`.

| Tier | Description | Entries |
|------|-------------|---------|
| **Tier 1** Synthetic | Known-true theorems with one condition removed | 59 |
| **Tier 2** Historical | Published conjectures with known counterexamples | 23 |
| **Tier 3** Subtle | Edge cases, imprecise statements, open questions | 10 |
| **Total** | | **92** |

- 73/92 entries (79.3 %) have verified ground-truth counterexamples
- Domains: number theory (69), inequality (13), combinatorics (10)
- Tier 2 includes: Euler sum-of-powers, Fermat primes, Mertens conjecture, Goldbach variants, Beal, Pólya, Legendre, twin primes, abc conjecture, and more

---

## Quick Start

### Install

```bash
git clone <repo-url>
cd conjlean
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Set your API key — only the provider you plan to use
# ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or HF_TOKEN
```

### Validate setup

```bash
make check    # 13-point environment validator
make smoke    # mock pipeline end-to-end test
```

### Run REFUTE on the benchmark

```bash
# Build the 3-tier benchmark dataset first
make build-benchmark

# Run REFUTE (uses provider from configs/config.yaml)
make refute-benchmark

# Evaluate results
make evaluate-refute
```

### Run ConjLean proof pipeline

```bash
# Build Lean + Mathlib (first time, ~30 min)
make lean

# Run with Anthropic (default)
python3 run.py run --config configs/config.yaml \
    --domains number_theory inequality --n-per-domain 100

# Evaluate
python3 run.py evaluate --results-dir data/results
```

---

## Lambda Labs Setup

### One-command setup

```bash
git clone <repo-url> conjlean && cd conjlean
bash scripts/setup_lambda.sh
```

Handles: system packages, Python deps, Lean 4 + Mathlib, validation.

### Inference (ConjLean / REFUTE)

| Setup | VRAM | Instance | Config |
|-------|------|----------|--------|
| 7B model (local HF) | 16 GB | 1× A100 40 GB | `config_lambda_7b.yaml` |
| 32B via vLLM | 80 GB | 1× A100 80 GB | vllm, port 8000 |
| 72B via vLLM | 160 GB | 2× A100 80 GB | `config_lambda_72b.yaml` |

```bash
# 7B model, single A100 — run REFUTE
python3 run.py refute \
    --config configs/config_lambda_7b.yaml \
    --benchmark-dir data/benchmark \
    --output data/refute_results_7b

# 72B via vLLM — start server first
bash scripts/start_vllm_lambda.sh Qwen/Qwen2.5-Math-72B-Instruct
make refute-72b   # (after server prints "Application startup complete")
```

### R-Agent Fine-tuning (DeepSeek-Math-7B via LoRA)

Trains the R-Agent on (conjecture, strategy) → (reasoning, counterexample) triples.
Requires a frontier LLM API key for data generation.

```bash
# Step 1 — generate supervised fine-tuning data from the benchmark
make gen-training-data   # needs ANTHROPIC_API_KEY (or equivalent)
# Output: data/training/samples.jsonl  (~3–5 traces × 73 CE entries)

# Step 2 — fine-tune on A100 (LoRA, rank 32, 4-bit quant)
make finetune-refuter
# Output: models/refuter_lora_v1/

# Step 3 — evaluate the fine-tuned R-Agent
python3 scripts/finetune_lora.py \
    --config configs/finetune_config.yaml \
    --data data/training/samples.jsonl \
    --output models/refuter_lora_v1 \
    --eval-only
```

**Fine-tuning config** (`configs/finetune_config.yaml`):
- Base model: `deepseek-ai/DeepSeek-Math-7B-Instruct`
- LoRA: rank 32, alpha 64, all attention + FFN projections
- Training: 3 epochs, LR 2e-4, cosine schedule, bf16
- Hardware: single A100 40 GB (4-bit), or A100 80 GB (full bf16)

Always use tmux on Lambda:
```bash
tmux new -s refute
# Detach: Ctrl+B D  |  Reattach: tmux attach -t refute
```

---

## CLI Reference

```bash
# ConjLean — prove conjectures
python3 run.py run \
    --config configs/config.yaml \
    --domains number_theory inequality \
    --n-per-domain 100

# ConjLean — evaluate results
python3 run.py evaluate --results-dir data/results

# ConjLean — formalize a single statement
python3 run.py formalize \
    --config configs/config.yaml \
    --statement "For all n, n*(n+1) is even"

# REFUTE — find counterexamples
python3 run.py refute \
    --config configs/config.yaml \
    --benchmark-dir data/benchmark \
    --output data/refute_results \
    --max-rounds 8 \
    --max-refinements 3 \
    --max-concurrent 5

# REFUTE — evaluate results
python3 run.py refute-evaluate \
    --results data/refute_results/loop_results.jsonl \
    --benchmark data/benchmark/all.jsonl

# List available providers
python3 run.py list-providers
```

---

## Makefile Reference

### Setup

| Target | Description |
|--------|-------------|
| `make install` | `pip install -e ".[dev]"` |
| `make install-lambda` | Install with `local_hf` extras for Lambda Labs |
| `make lean` | Build Lean 4 + Mathlib |
| `make lean-update` | `lake update && lake build` |

### Quality

| Target | Description |
|--------|-------------|
| `make test` | Run all 225 tests |
| `make check` | 13-point environment validation |
| `make smoke` | End-to-end mock pipeline test |
| `make lint` | `ruff check` |
| `make fmt` | `ruff format` |

### ConjLean Pipeline

| Target | Description |
|--------|-------------|
| `make run-small` | 50 conjectures/domain, 7B model |
| `make run-7b` | 200 conjectures/domain, 7B model |
| `make run-72b` | 200 conjectures/domain, 72B via vLLM |
| `make evaluate` | Evaluate latest results in `data/results` |

### REFUTE Pipeline

| Target | Description |
|--------|-------------|
| `make build-benchmark` | Build all 3 tiers → `data/benchmark/` |
| `make gen-training-data` | Generate LoRA SFT data (needs API key) |
| `make finetune-refuter` | Fine-tune DeepSeek-Math-7B via LoRA |
| `make refute-benchmark` | Run REFUTE on full benchmark |
| `make refute-7b` | Run REFUTE with fine-tuned 7B model |
| `make evaluate-refute` | Evaluate REFUTE results |

---

## Project Structure

```
conjlean/
├── src/                              # Core library (importable as `conjlean`)
│   ├── schemas.py                    # All dataclasses and enums (shared)
│   ├── config.py                     # Pydantic v2 config (all settings)
│   ├── models.py                     # LLM client abstraction (6 providers)
│   │
│   ├── # ConjLean — proof pipeline
│   ├── conjecture_gen.py             # LLM conjecture generation
│   ├── sympy_filter.py               # Fast SymPy pre-checker
│   ├── formalizer.py                 # NL → Lean 4 with error-repair loop
│   ├── lean_harness.py               # Lean 4 REPL subprocess wrapper
│   ├── proof_search.py               # 4-layer tactic proof search
│   ├── pipeline.py                   # ConjLean end-to-end orchestration
│   ├── evaluate.py                   # ConjLean metrics and report
│   │
│   └── # REFUTE — counterexample pipeline
│       ├── refuter.py                # R-Agent: 4 counterexample strategies
│       ├── strategist.py             # S-Agent: UCB1 meta-controller
│       ├── refute_loop.py            # Full REFUTE loop orchestration
│       ├── benchmark.py              # 3-tier benchmark builder + loader
│       └── refute_evaluate.py        # REFUTE metrics, LaTeX table export
│
├── scripts/
│   ├── build_benchmark.py            # CLI: build data/benchmark/
│   ├── gen_training_data.py          # CLI: generate LoRA SFT data
│   ├── finetune_lora.py              # CLI: fine-tune DeepSeek-Math-7B
│   ├── check_setup.py                # 13-point environment validator
│   ├── run_smoke_test.py             # Mock pipeline smoke test
│   ├── setup_lambda.sh               # One-shot Lambda Labs setup
│   └── start_vllm_lambda.sh          # Start vLLM server (32B/72B)
│
├── configs/
│   ├── config.yaml                   # Default (Anthropic Claude)
│   ├── config_lambda_7b.yaml         # Lambda Labs, 1× A100, 7B local HF
│   ├── config_lambda_72b.yaml        # Lambda Labs, 2× A100, 72B vLLM
│   └── finetune_config.yaml          # LoRA fine-tuning (DeepSeek-Math-7B)
│
├── lean/                             # Lean 4 project (Mathlib dependency)
│   ├── lakefile.toml
│   └── ConjLean/Basic.lean
│
├── tests/                            # 225 pytest tests
│   ├── test_refuter.py               # R-Agent unit tests
│   ├── test_refute_loop.py           # REFUTE loop unit tests
│   ├── test_schemas.py
│   ├── test_config.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   ├── test_formalizer.py
│   ├── test_lean_harness.py
│   ├── test_proof_search.py
│   └── test_sympy_filter.py
│
├── data/
│   ├── benchmark/                    # Generated by make build-benchmark
│   │   ├── tier1.jsonl               # 59 synthetic falsehoods
│   │   ├── tier2.jsonl               # 23 historical conjectures
│   │   ├── tier3.jsonl               # 10 subtle / open cases
│   │   └── all.jsonl                 # All 92 entries combined
│   ├── training/                     # Generated by make gen-training-data
│   └── results/                      # Generated by pipeline runs
│
├── models/                           # Fine-tuned LoRA adapters (gitignored)
├── run.py                            # CLI entry point
├── setup.py                          # Package definition
├── Makefile
└── .env.example
```

---

## Supported LLM Providers

| Provider | Config key | Typical model |
|----------|-----------|---------------|
| Anthropic | `anthropic` | `claude-sonnet-4-6` |
| OpenAI | `openai` | `gpt-4o`, `o1` |
| Google Gemini | `gemini` | `gemini-1.5-pro` |
| HuggingFace API | `huggingface` | Inference API, no GPU |
| vLLM | `vllm` | Any GGUF/HF model, self-hosted |
| Local HF | `local_hf` | Direct `transformers`, Lambda Labs |

---

## Paper

**REFUTE: Property-Based Testing for Mathematics via Multi-Agent Counterexample Search**
ICML AI4Research Workshop 2026

**Framing**: We reframe automated mathematical reasoning as a *property-based testing* problem. Instead of asking "can we prove this?", we ask "can we find a value that breaks it?". The REFUTE system uses a 4-agent loop to systematically search for counterexamples, refine conjectures when counterexamples are found, and learn which search strategies work best per domain.

**Key contributions**:
1. **REFUTE loop** — C/R/V/S agent architecture with UCB1-guided strategy selection
2. **3-tier benchmark** — 92 curated conjectures (synthetic, historical, subtle) with verified ground-truth
3. **LoRA R-Agent** — DeepSeek-Math-7B fine-tuned on frontier-generated reasoning traces
4. **Empirical evaluation** — precision/recall/F1 breakdowns by tier, domain, and strategy; ablation over each agent component

**Baseline comparison**: API-only (Claude Sonnet, no fine-tuning) vs. LoRA R-Agent (7B, ~5k training triples from benchmark).
