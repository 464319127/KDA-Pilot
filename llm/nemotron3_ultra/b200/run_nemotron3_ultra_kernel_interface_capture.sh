#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}

export ROOT
export MODEL_SLUG=${MODEL_SLUG:-nemotron3_ultra}
export MODEL=${MODEL:-nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TP_SIZE=${TP_SIZE:-8}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.95}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-8192}
export MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-32}
export HIGH_PROMPTS=${HIGH_PROMPTS:-32}
export HIGH_CONCURRENCY=${HIGH_CONCURRENCY:-32}
export SHAREGPT_CONTEXT_LEN=${SHAREGPT_CONTEXT_LEN:-8192}
export SERVER_ARGS_EXTRA=${SERVER_ARGS_EXTRA:---trust-remote-code --context-length 8192 --max-total-tokens 65536 --max-prefill-tokens 8192 --kv-cache-dtype fp8_e4m3 --mamba-radix-cache-strategy extra_buffer --moe-runner-backend triton --ep-size 8 --enforce-disable-flashinfer-allreduce-fusion --disable-flashinfer-autotune --skip-server-warmup --enable-memory-saver}

exec "${ROOT}/scripts/run_kernel_interface_capture.sh"
