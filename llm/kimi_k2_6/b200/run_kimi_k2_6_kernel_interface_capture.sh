#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
export ROOT

export MODEL_SLUG=${MODEL_SLUG:-kimi_k2_6}
export MODEL=${MODEL:-moonshotai/Kimi-K2.6}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TP_SIZE=${TP_SIZE:-8}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-32768}
export MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-80}
export DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-1}
if [[ -z "${SERVER_ARGS_EXTRA:-}" ]]; then
  SERVER_ARGS_EXTRA='--trust-remote-code --context-length 128000 --reasoning-parser kimi_k2 --tool-call-parser kimi_k2 --attention-backend triton --enforce-disable-flashinfer-allreduce-fusion'
fi
export SERVER_ARGS_EXTRA

exec "${ROOT}/scripts/run_kernel_interface_capture.sh"
