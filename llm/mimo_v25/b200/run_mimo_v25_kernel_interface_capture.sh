#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}

export ROOT
export MODEL_SLUG=${MODEL_SLUG:-mimo_v25}
export MODEL=${MODEL:-XiaomiMiMo/MiMo-V2.5}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TP_SIZE=${TP_SIZE:-4}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.45}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-16384}
export MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-80}
export DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-1}
export SERVER_ARGS_EXTRA=${SERVER_ARGS_EXTRA:---trust-remote-code --reasoning-parser mimo --tool-call-parser mimo --attention-backend triton --mm-attention-backend fa4 --swa-full-tokens-ratio 0.1 --enable-memory-saver --disable-flashinfer-autotune --enforce-disable-flashinfer-allreduce-fusion}

exec "${ROOT}/scripts/run_kernel_interface_capture.sh"
