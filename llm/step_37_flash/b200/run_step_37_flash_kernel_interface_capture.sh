#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}

export ROOT
export MODEL_SLUG=${MODEL_SLUG:-step_37_flash}
export MODEL=${MODEL:-stepfun-ai/Step-3.7-Flash-FP8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TP_SIZE=${TP_SIZE:-8}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-32768}
export MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-80}
export DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-0}
export SERVER_ARGS_EXTRA=${SERVER_ARGS_EXTRA:---ep 8 --moe-runner-backend triton --attention-backend triton --trust-remote-code --reasoning-parser step3p5 --tool-call-parser step3p5 --disable-flashinfer-autotune --enforce-disable-flashinfer-allreduce-fusion --disable-cuda-graph --disable-piecewise-cuda-graph}

exec "${ROOT}/scripts/run_kernel_interface_capture.sh"
