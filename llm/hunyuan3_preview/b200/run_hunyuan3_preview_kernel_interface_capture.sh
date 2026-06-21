#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}

export ROOT
export MODEL_SLUG=${MODEL_SLUG:-hunyuan3_preview}
export MODEL=${MODEL:-tencent/Hy3-preview}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TP_SIZE=${TP_SIZE:-8}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.9}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-32768}
export MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-80}
export DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-1}
export SERVER_ARGS_EXTRA=${SERVER_ARGS_EXTRA:---trust-remote-code --reasoning-parser hunyuan --tool-call-parser hunyuan --attention-backend triton --disable-flashinfer-autotune --enforce-disable-flashinfer-allreduce-fusion}

exec "${ROOT}/scripts/run_kernel_interface_capture.sh"
