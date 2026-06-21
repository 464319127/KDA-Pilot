#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
export ROOT

export MODEL_SLUG=${MODEL_SLUG:-minimax_m3}
export MODEL=${MODEL:-MiniMaxAI/MiniMax-M3-MXFP8}
export SGLANG_REPO=${SGLANG_REPO:-/sgl-workspace/sglang}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TP_SIZE=${TP_SIZE:-8}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.65}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
export MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-80}
export DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-1}
export SGLANG_DISABLE_MSA=${SGLANG_DISABLE_MSA:-1}
if [[ -z "${SERVER_ARGS_EXTRA:-}" ]]; then
  SERVER_ARGS_EXTRA='--trust-remote-code --reasoning-parser auto --tool-call-parser auto --attention-backend fa4 --page-size 128 --moe-runner-backend deep_gemm'
fi
export SERVER_ARGS_EXTRA

exec "${ROOT}/scripts/run_kernel_interface_capture.sh"
