#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
export ROOT

export MODEL_SLUG=${MODEL_SLUG:-ring_26_1t}
export MODEL=${MODEL:-inclusionAI/Ring-2.6-1T}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TP_SIZE=${TP_SIZE:-8}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-32768}
export MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-80}
export DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-1}
if [[ -z "${SERVER_ARGS_EXTRA:-}" ]]; then
  SERVER_ARGS_EXTRA='--trust-remote-code --model-loader-extra-config {"enable_multithread_load":"true","num_threads":64} --tool-call-parser glm --reasoning-parser deepseek-r1'
fi
export SERVER_ARGS_EXTRA

exec "${ROOT}/scripts/run_kernel_interface_capture.sh"
