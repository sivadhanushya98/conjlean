# ConjLean

**Automated mathematical conjecture generation and formal verification using LLMs + Lean 4.**

ConjLean is an end-to-end pipeline that:
1. **Generates** mathematical conjectures using an LLM
2. **Filters** them with SymPy (fast CPU pre-check)
3. **Formalizes** natural-language statements into Lean 4 with an error-repair loop
4. **Proves** them with a 4-layer tactic waterfall + LLM-guided proof search

All verified proofs are formally checked by Lean 4 / Mathlib — no human mathematician required.

---

## Architecture

```
LLM  →  SymPy filter  →  Lean 4 autoformalization  →  4-layer proof search
  Generate          Pre-check           NL → Lean4               Verify
  conjectures       (5 s, CPU)          (repair loop)            Layer 0: auto tactics
                                                                 Layer 1: tactic combos
                                                                 Layer 2: exact?/apply?
                                                                 Layer 3: LLM + Lean REPL
```

**Domains**: Number theory, inequalities (where `omega`, `norm_num`, `nlinarith`, `positivity` are strong).

---

## Supported Providers

| Provider | Config key | Notes |
|---|---|---|
| Anthropic | `anthropic` | Claude Sonnet / Opus |
| OpenAI | `openai` | GPT-4o, o1, etc. |
| Google Gemini | `gemini` | Gemini 1.5 Pro |
| HuggingFace API | `huggingface` | Inference API, no GPU needed |
| vLLM | `vllm` | Self-hosted OpenAI-compatible server |
| Local HF | `local_hf` | Direct `transformers` inference, best for Lambda Labs |

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd conjlean
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API key (only the provider you plan to use)
```

### 3. Build Lean / Mathlib (first time only, ~30 min)

```bash
# Install elan (Lean version manager)
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | bash -s -- -y

export PATH="$HOME/.elan/bin:$PATH"
cd lean && lake build && cd ..
```

### 4. Validate setup

```bash
make check
make smoke
```

### 5. Run the pipeline

```bash
# Default (Anthropic, 100 conjectures per domain)
python3 run.py run --config configs/config.yaml \
    --domains number_theory inequality --n-per-domain 100

# Evaluate results
python3 run.py evaluate --results-dir data/results
```

---

## Lambda Labs Setup

This is designed to run efficiently on [Lambda Labs](https://lambdalabs.com) GPU instances.

### One-command setup

```bash
# SSH into your Lambda instance, then:
git clone <repo-url> conjlean && cd conjlean
bash scripts/setup_lambda.sh
```

The setup script handles: system packages, Python deps, Lean 4 + Mathlib build, and validation.

### Option A: 7B model (single A100, no API key)

Works on any Lambda Labs A100 instance (40 GB or 80 GB).

```bash
make run-small
# or
python3 run.py run --config configs/config_lambda_7b.yaml \
    --domains number_theory inequality --n-per-domain 100
```

### Option B: 72B model via vLLM (2× A100 SXM4 80 GB)

```bash
# Terminal 1 — start vLLM server
bash scripts/start_vllm_lambda.sh Qwen/Qwen2.5-Math-72B-Instruct

# Terminal 2 — run pipeline (after server prints "Application startup complete")
make run-72b
```

### Recommended Lambda instances

| Model | VRAM | Instance | Config |
|---|---|---|---|
| Qwen2.5-Math-7B-Instruct | 16 GB | 1× A100 SXM4 40GB | `config_lambda_7b.yaml` |
| Qwen/QwQ-32B | 80 GB | 1× A100 SXM4 80GB | vllm, port 8000 |
| Qwen2.5-Math-72B-Instruct | 160 GB | 2× A100 SXM4 80GB | `config_lambda_72b.yaml` |

### Use tmux on Lambda

Always run long jobs in tmux so they survive SSH disconnects:

```bash
tmux new -s conjlean
# ... run your commands ...
# Detach: Ctrl+B, D
# Reattach: tmux attach -t conjlean
```

---

## Project Structure

```
conjlean/
├── src/                        # Core pipeline modules
│   ├── config.py               # Pydantic v2 config (all settings)
│   ├── schemas.py              # Shared dataclasses and enums
│   ├── models.py               # LLM client abstraction (all providers)
│   ├── sympy_filter.py         # Fast SymPy pre-checker
│   ├── formalizer.py           # NL → Lean 4 with error-repair loop
│   ├── lean_harness.py         # Lean 4 REPL subprocess wrapper
│   ├── proof_search.py         # 4-layer proof search
│   ├── pipeline.py             # End-to-end orchestration
│   └── evaluate.py             # Metrics and evaluation report
├── lean/                       # Lean 4 project (Mathlib)
│   ├── lakefile.toml
│   └── ConjLean/Basic.lean
├── configs/
│   ├── config.yaml             # Default config (Anthropic)
│   ├── config_lambda_7b.yaml   # Lambda Labs, single A100, 7B
│   └── config_lambda_72b.yaml  # Lambda Labs, 2× A100 80G, 72B vLLM
├── tests/                      # 209 unit tests (pytest)
├── scripts/
│   ├── setup_lambda.sh         # One-shot Lambda Labs setup
│   ├── start_vllm_lambda.sh    # Start vLLM server for 32B/72B models
│   ├── check_setup.py          # 13-point environment validator
│   └── run_smoke_test.py       # Full mock pipeline test
├── run.py                      # CLI entry point
├── setup.py                    # Package definition
├── Makefile                    # Common targets
└── .env.example                # API key template
```

---

## CLI Reference

```bash
# Run pipeline
python3 run.py run --config configs/config.yaml \
    --domains number_theory inequality \
    --n-per-domain 100

# Evaluate results
python3 run.py evaluate --results-dir data/results

# Formalize a single statement
python3 run.py formalize --config configs/config.yaml \
    --statement "For all n, n*(n+1) is even"

# List available providers
python3 run.py list-providers
```

---

## Makefile Targets

| Target | Description |
|---|---|
| `make install` | `pip install -e ".[dev]"` |
| `make install-lambda` | Install with local_hf extras |
| `make test` | Run all 209 tests |
| `make check` | 13-point setup validation |
| `make smoke` | Mock pipeline smoke test |
| `make lean` | Build Lean + Mathlib |
| `make run-small` | 50 conjectures per domain (7B) |
| `make run-7b` | 200 conjectures per domain (7B) |
| `make run-72b` | 200 conjectures per domain (72B vLLM) |
| `make evaluate` | Evaluate latest results |
| `make clean` | Remove build artifacts |

---

## Paper

This pipeline is being used for a paper submission to **AI4Research @ ICML 2026**.

**Framing**: ML systems paper — we characterize what a no-fine-tuning LLM + Lean 4 pipeline
can automatically discover and verify, and analyze where it succeeds and fails.
Lean is the mathematician. No human verification required.

**Claims**:
- System description and design decisions
- Empirical characterization across number theory and inequality domains
- Failure mode analysis (formalization errors, proof layer breakdown)
- Comparison across model sizes (7B vs 72B)
