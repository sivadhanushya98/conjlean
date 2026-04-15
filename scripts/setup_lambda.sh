#!/usr/bin/env bash
# setup_lambda.sh — one-shot Lambda Labs instance setup for ConjLean
#
# Run this once after SSHing into a fresh Lambda Labs GPU instance:
#   bash scripts/setup_lambda.sh
#
# What it does:
#   1. System update + essential packages
#   2. Python dependencies (pip install -e .[dev,local_hf])
#   3. Lean 4 + Mathlib (elan + lake build — takes 20-40 min first time)
#   4. Sanity checks via scripts/check_setup.py
#
# Optional flags:
#   --skip-lean      Skip Lean 4 / Mathlib build (if already installed)
#   --skip-pip       Skip Python package installation
#   --download-7b    Pre-download Qwen2.5-Math-7B-Instruct weights now
#   --download-72b   Pre-download Qwen2.5-Math-72B-Instruct weights now

set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
step()  { echo -e "\n${CYAN}══ $* ${NC}"; }
info()  { echo -e "${GREEN}[OK]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERR]${NC}  $*" >&2; exit 1; }

# ── Argument parsing ───────────────────────────────────────────────────────────
SKIP_LEAN=false
SKIP_PIP=false
DOWNLOAD_7B=false
DOWNLOAD_72B=false

for arg in "$@"; do
    case $arg in
        --skip-lean)     SKIP_LEAN=true ;;
        --skip-pip)      SKIP_PIP=true ;;
        --download-7b)   DOWNLOAD_7B=true ;;
        --download-72b)  DOWNLOAD_72B=true ;;
        *) warn "Unknown flag: $arg (ignored)" ;;
    esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
info "Working directory: $REPO_DIR"

# ── 1. GPU check ──────────────────────────────────────────────────────────────
step "GPU environment"
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    info "Detected ${GPU_COUNT} GPU(s):"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
        | while IFS= read -r line; do echo "    GPU $line"; done
else
    warn "nvidia-smi not found — CPU mode only (slow for large models)"
fi

# ── 2. System packages ─────────────────────────────────────────────────────────
step "System packages"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    curl wget git tmux htop \
    build-essential cmake \
    ca-certificates \
    2>/dev/null || warn "Some apt packages may have failed — continuing"
info "System packages OK"

# ── 3. Python dependencies ─────────────────────────────────────────────────────
if [[ "$SKIP_PIP" == "false" ]]; then
    step "Python dependencies"
    python3 --version
    pip install --upgrade pip --quiet
    pip install -e ".[dev,local_hf]" --quiet
    info "Python dependencies installed"
else
    info "Skipping pip install (--skip-pip)"
fi

# ── 4. Lean 4 + Mathlib ────────────────────────────────────────────────────────
if [[ "$SKIP_LEAN" == "false" ]]; then
    step "Lean 4 + elan"

    # Install elan (Lean version manager)
    if ! command -v elan &>/dev/null; then
        info "Installing elan..."
        curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
            | bash -s -- -y --default-toolchain none
        export PATH="$HOME/.elan/bin:$PATH"
        info "elan installed"
    else
        info "elan already installed: $(elan --version 2>/dev/null || echo 'version unknown')"
    fi

    # Ensure elan is on PATH for this session
    export PATH="$HOME/.elan/bin:$PATH"

    step "Mathlib lake build (this takes 20-40 min on first run)"
    echo ""
    warn "Building Mathlib from source. Do NOT interrupt — use tmux to persist if needed."
    warn "Start tmux before running this script:  tmux new -s setup"
    echo ""

    cd "$REPO_DIR/lean"
    lake update
    lake build

    cd "$REPO_DIR"
    info "Lean 4 + Mathlib build complete"
else
    info "Skipping Lean build (--skip-lean)"
fi

# ── 5. Pre-download model weights (optional) ───────────────────────────────────
if [[ "$DOWNLOAD_7B" == "true" ]]; then
    step "Pre-downloading Qwen2.5-Math-7B-Instruct (~16 GB)"
    python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_id = 'Qwen/Qwen2.5-Math-7B-Instruct'
print(f'Downloading tokenizer...')
AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(f'Downloading model weights (bfloat16)...')
AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
print('Done.')
"
    info "Qwen2.5-Math-7B-Instruct cached"
fi

if [[ "$DOWNLOAD_72B" == "true" ]]; then
    step "Pre-downloading Qwen2.5-Math-72B-Instruct (~160 GB, this will take a while)"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-Math-72B-Instruct')
print('Done.')
"
    info "Qwen2.5-Math-72B-Instruct cached"
fi

# ── 6. Final setup check ────────────────────────────────────────────────────────
step "Setup validation"
python3 scripts/check_setup.py || warn "Some checks failed — see output above"

# ── 7. Print next steps ────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ConjLean setup complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Next steps:"
echo ""
echo "  Option A — 7B model (single GPU, no API key needed):"
echo "    make run-small"
echo "    # or: python3 run.py run --config configs/config_lambda_7b.yaml \\"
echo "    #         --domains number_theory inequality --n-per-domain 100"
echo ""
echo "  Option B — 72B model via vLLM (2× A100 80G):"
echo "    # Terminal 1:"
echo "    bash scripts/start_vllm_lambda.sh Qwen/Qwen2.5-Math-72B-Instruct"
echo "    # Terminal 2 (after server is ready):"
echo "    make run-72b"
echo ""
echo "  Option C — use an API (Anthropic / OpenAI / Gemini):"
echo "    cp .env.example .env"
echo "    # Edit .env with your API key"
echo "    python3 run.py run --config configs/config.yaml \\"
echo "        --provider anthropic --domains number_theory --n-per-domain 100"
echo ""
echo "  Evaluate results:"
echo "    make evaluate"
echo ""
