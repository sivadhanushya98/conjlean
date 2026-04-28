.PHONY: install test lean smoke check run-small run-full clean fmt lint

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

install-lambda:
	pip install -e ".[dev,local_hf]"

# ── Quality ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -x -q

lint:
	ruff check src/ tests/ run.py

fmt:
	ruff format src/ tests/ run.py

# ── Lean / Mathlib ────────────────────────────────────────────────────────────
lean:
	cd lean && lake build

lean-update:
	cd lean && lake update && lake build

# ── Validation ────────────────────────────────────────────────────────────────
check:
	python3 scripts/check_setup.py

smoke:
	python3 scripts/run_smoke_test.py

# ── Pipeline runs ─────────────────────────────────────────────────────────────
# Small run — 7B model via local HuggingFace (single A100 40/80 GB)
run-small:
	python3 run.py run \
		--config configs/config_lambda_7b.yaml \
		--domains number_theory inequality \
		--n-per-domain 50

# Full run — 7B model, larger batch
run-7b:
	python3 run.py run \
		--config configs/config_lambda_7b.yaml \
		--domains number_theory inequality \
		--n-per-domain 200

# Full run — 72B via vLLM (start vLLM server first: bash scripts/start_vllm_lambda.sh)
run-72b:
	python3 run.py run \
		--config configs/config_lambda_72b.yaml \
		--domains number_theory inequality \
		--n-per-domain 200

# Evaluate most recent results
evaluate:
	python3 run.py evaluate --results-dir data/results

# ── REFUTE — Counterexample pipeline ─────────────────────────────────────────

# Build the 3-tier benchmark dataset
build-benchmark:
	python3 scripts/build_benchmark.py \
		--validate \
		--output-dir data/benchmark

# Generate LoRA training data (requires ANTHROPIC_API_KEY or similar)
gen-training-data:
	python3 scripts/gen_training_data.py \
		--config configs/config.yaml \
		--benchmark-dir data/benchmark \
		--output data/training/samples.jsonl \
		--n-traces 3 \
		--max-concurrent 5

# Fine-tune DeepSeek-Math-7B as the R-Agent via LoRA
finetune-refuter:
	python3 scripts/finetune_lora.py \
		--config configs/finetune_config.yaml \
		--data data/training/samples.jsonl \
		--output models/refuter_lora_v1

# Run REFUTE on the full benchmark (API-based, uses config.yaml provider)
refute-benchmark:
	python3 run.py refute \
		--config configs/config.yaml \
		--benchmark-dir data/benchmark \
		--output data/refute_results

# Run REFUTE with fine-tuned 7B model (Lambda Labs)
refute-7b:
	python3 run.py refute \
		--config configs/config_lambda_7b.yaml \
		--benchmark-dir data/benchmark \
		--output data/refute_results_7b

# Evaluate REFUTE results
evaluate-refute:
	python3 run.py refute-evaluate \
		--results data/refute_results/loop_results.jsonl \
		--benchmark data/benchmark/all.jsonl

# ── Utilities ─────────────────────────────────────────────────────────────────
list-providers:
	python3 run.py list-providers

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/
