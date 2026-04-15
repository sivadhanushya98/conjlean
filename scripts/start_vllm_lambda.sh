#!/usr/bin/env bash
# start_vllm_lambda.sh — spin up a vLLM OpenAI-compatible server on Lambda Labs
#
# Usage:
#   bash scripts/start_vllm_lambda.sh [MODEL] [PORT] [GPUS]
#
# Examples:
#   bash scripts/start_vllm_lambda.sh                                  # 7B, port 8000, all GPUs
#   bash scripts/start_vllm_lambda.sh Qwen/Qwen2.5-Math-72B-Instruct  # 72B (needs 2×A100-80G)
#   bash scripts/start_vllm_lambda.sh Qwen/QwQ-32B 8001               # 32B on port 8001
#
# After the server starts, update configs/config.yaml:
#   provider: vllm
#   vllm:
#     base_url: "http://localhost:<PORT>/v1"
#     model: "<MODEL>"
#
# Then run:
#   python3 run.py run --config configs/config.yaml --provider vllm ...

set -euo pipefail

# ── Arguments ────────────────────────────────────────────────────────────────
MODEL="${1:-Qwen/Qwen2.5-Math-7B-Instruct}"
PORT="${2:-8000}"
TENSOR_PARALLEL="${3:-}"   # leave empty → vLLM auto-detects GPU count

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[vLLM]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERR ]${NC} $*" >&2; exit 1; }

# ── Recommended model reference ───────────────────────────────────────────────
info "Recommended math models for ConjLean:"
echo ""
echo "  Model                                    VRAM    Lambda instance"
echo "  ─────────────────────────────────────────────────────────────────"
echo "  Qwen/Qwen2.5-Math-7B-Instruct           16 GB   1× A100 SXM4 40GB"
echo "  deepseek-ai/DeepSeek-Math-7B-Instruct   16 GB   1× A100 SXM4 40GB"
echo "  Qwen/QwQ-32B                             80 GB   1× A100 SXM4 80GB"
echo "  Qwen/Qwen2.5-Math-72B-Instruct          160 GB  2× A100 SXM4 80GB"
echo "  meta-llama/Meta-Llama-3.1-70B-Instruct  160 GB  2× A100 SXM4 80GB"
echo ""
info "Starting: model=${MODEL}  port=${PORT}"

# ── Check CUDA ────────────────────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    warn "nvidia-smi not found — continuing anyway (CPU mode will be slow)"
else
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    info "Detected ${GPU_COUNT} GPU(s):"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
        | awk '{print "    GPU " $0}'
    echo ""
    if [[ -z "${TENSOR_PARALLEL}" ]]; then
        TENSOR_PARALLEL="${GPU_COUNT}"
        info "Auto tensor-parallel: ${TENSOR_PARALLEL}"
    fi
fi

# ── Install vLLM if needed ────────────────────────────────────────────────────
if ! python3 -c "import vllm" &>/dev/null; then
    info "vLLM not found — installing (this may take a few minutes)..."
    pip install vllm --quiet
    info "vLLM installed."
fi

VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
info "vLLM version: ${VLLM_VERSION}"

# ── HuggingFace login (optional — needed for gated models like Llama-3) ───────
if [[ -n "${HF_TOKEN:-}" ]]; then
    info "HF_TOKEN set — authenticating with HuggingFace Hub..."
    python3 -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" \
        && info "HF login OK" || warn "HF login failed — gated models may not download"
else
    info "HF_TOKEN not set — public models only (Qwen, DeepSeek are public)"
fi

# ── Build the vLLM command ────────────────────────────────────────────────────
VLLM_CMD=(
    python3 -m vllm.entrypoints.openai.api_server
    --model "${MODEL}"
    --port "${PORT}"
    --host "0.0.0.0"
    --dtype "bfloat16"
    --trust-remote-code
    --max-model-len 8192
    --gpu-memory-utilization 0.92
)

if [[ -n "${TENSOR_PARALLEL}" && "${TENSOR_PARALLEL}" -gt 1 ]]; then
    VLLM_CMD+=(--tensor-parallel-size "${TENSOR_PARALLEL}")
fi

# ── Print connection instructions ─────────────────────────────────────────────
echo ""
info "Server will be available at: http://localhost:${PORT}/v1"
echo ""
echo "  Update configs/config.yaml:"
echo "    provider: vllm"
echo "    vllm:"
echo "      base_url: \"http://localhost:${PORT}/v1\""
echo "      model: \"${MODEL}\""
echo ""
echo "  Then run:"
echo "    python3 run.py run --config configs/config.yaml --provider vllm \\"
echo "        --domains number_theory inequality --n-per-domain 100"
echo ""
info "Starting vLLM server... (Ctrl+C to stop)"
echo ""

# ── Launch ───────────────────────────────────────────────────────────────────
exec "${VLLM_CMD[@]}"
