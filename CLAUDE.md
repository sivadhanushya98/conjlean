# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ConjLean** is an ICML AI4Research 2026 submission implementing two complementary automated mathematical reasoning systems:

1. **ConjLean** — Generate conjectures → SymPy symbolic filter → Lean 4 formalization → 4-layer proof search
2. **REFUTE** — Multi-agent counterexample search (R-Agent strategies + S-Agent UCB1 selection + C-Agent refinement)

## Commands

### Setup
```bash
make install          # pip install -e ".[dev]"
make install-lambda   # includes local_hf extras (PEFT, TRL, bitsandbytes)
make lean             # lake build (Lean 4 + Mathlib v4.14.0)
make lean-update      # lake update && lake build
make check            # 13-point environment validation
```

### Development
```bash
make test             # pytest tests/ (225+ tests)
make lint             # ruff check src/ tests/ run.py
make fmt              # ruff format src/ tests/ run.py
make smoke            # end-to-end mock pipeline test
```

### Running a single test file or test
```bash
pytest tests/test_refuter.py
pytest tests/test_pipeline.py::test_full_pipeline_run -v
pytest -k "test_boundary" tests/
```

### ConjLean pipeline
```bash
python3 run.py run --config configs/config.yaml --domains number_theory inequality --n-per-domain 100
python3 run.py evaluate --results-dir data/results
python3 run.py formalize --config configs/config.yaml --statement "For all n, n*(n+1) is even"
```

### REFUTE pipeline
```bash
make build-benchmark              # generates data/benchmark/{tier1,tier2,tier3,all}.jsonl
python3 run.py refute --config configs/config.yaml --benchmark-dir data/benchmark --output data/refute_results --max-rounds 8
python3 run.py refute-evaluate --results data/refute_results/loop_results.jsonl --benchmark data/benchmark/all.jsonl
```

### LoRA fine-tuning (Lambda Labs)
```bash
make gen-training-data    # generates data/training/samples.jsonl (3–5k SFT triples)
make finetune-refuter     # trains LoRA on DeepSeek-Math-7B-Instruct → models/refuter_lora_v1/
make refute-7b            # run REFUTE with fine-tuned 7B
```

## Architecture

### Module Layout (`src/`)

All source lives in `src/` and is importable as `conjlean`.

**Shared infrastructure:**
- `schemas.py` — All dataclasses (`Conjecture`, `FilterResult`, `FormalizedConjecture`, `ProofResult`, `PipelineResult`, `RefuteLoopResult`, etc.)
- `config.py` — Pydantic v2 config loaded from YAML; env vars override keys
- `models.py` — 6 LLM provider clients (`AnthropicClient`, `OpenAIClient`, `GeminiClient`, `HuggingFaceClient`, `LocalHFClient`, vLLM via OpenAI-compatible base_url override)

**ConjLean pipeline** (proof-oriented, runs stages in sequence via `pipeline.py`):
- `conjecture_gen.py` → batch LLM calls (async, temperature 0.8, batch 20)
- `sympy_filter.py` → multiprocessing pool with 5s SIGALRM timeout; conservatively fails only on confirmed counterexamples
- `formalizer.py` → NL→Lean 4 with error-repair loop (max 5 retries via `lean_harness.py`)
- `lean_harness.py` → long-running Lean REPL subprocess; JSON protocol for commands/responses
- `proof_search.py` → 4 layers with escalating timeouts: Layer 0 `auto` (5s), Layer 1 tactic combos (30s), Layer 2 search tactics (60s), Layer 3 LLM dialogue (120s)

**REFUTE pipeline** (counterexample-oriented):
- `refuter.py` (R-Agent) — 4 strategies: `BOUNDARY` (deterministic sweep), `RANDOM_STRUCTURED` (primes/squares/Fibonacci), `ANALOGICAL` (LLM-proposed from past CEs), `SYMBOLIC_PERTURBATION` (LLM-identified parameter perturbation); SymPy verification with 4s async timeout
- `strategist.py` (S-Agent) — UCB1 bandit (sqrt(2) constant) over strategies; LLM fallback for cold start (< 5 samples); terminates if best win-rate < 5%
- `refute_loop.py` — Orchestrates S→R→V→C cycle; C-Agent proposes refined statements on confirmed CEs (max 3 refinements); incremental JSONL save per conjecture

### Key Design Patterns

**Async-first**: `asyncio.gather()` for all batch LLM calls throughout the codebase. Tests use `pytest-asyncio` with `AsyncMock`.

**Incremental JSONL checkpointing**: Both `pipeline.py` and `refute_loop.py` append per-conjecture results to JSONL on completion so crashes don't lose progress.

**Subprocess isolation**: SymPy filter runs in `multiprocessing` worker pool (not threads); Lean REPL is a persistent subprocess communicating via newline-delimited JSON.

**Pydantic v2 config**: Strongly typed YAML config + env var overrides. Use `config.py` to add new config fields — do not hardcode parameters.

**LLM abstraction**: All LLM calls go through the `LLMClient` abstract base. Swap providers by changing `config.yaml` — no code changes needed.

### Config Files

| File | Provider | Model |
|------|----------|-------|
| `configs/config.yaml` | Anthropic (default) | claude-sonnet-4-6 |
| `configs/config_lambda_7b.yaml` | Local HF | Qwen/Qwen2.5-Math-7B-Instruct |
| `configs/config_lambda_72b.yaml` | vLLM | Qwen/Qwen2.5-Math-72B-Instruct |
| `configs/finetune_config.yaml` | — | DeepSeek-Math-7B-Instruct (LoRA training) |

### Benchmark Structure

`data/benchmark/` contains 92 entries across 3 tiers (built via `make build-benchmark`):
- **Tier 1** (59): Synthetically-false conjectures; all have programmatically-verified counterexamples
- **Tier 2** (23): Historical conjectures with known status (Euler, Fermat, Mertens, Goldbach, etc.)
- **Tier 3** (10): Subtle/open cases testing scope ambiguity detection

### Test Fixtures (`tests/conftest.py`)

- `sample_conjecture(domain)` — factory for `Conjecture` objects
- `mock_llm_client()` — `AsyncMock` `LLMClient` with controllable responses
- `config_path` — resolved path to `configs/config.yaml`

## Environment

Copy `.env.example` to `.env` and fill in API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`). The Lean 4 REPL requires `lake` on PATH — run `make lean` after setup.
