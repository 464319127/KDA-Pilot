#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}

export ROOT
export MODEL_SLUG=${MODEL_SLUG:-deepseek_v4}
export MODEL=${MODEL:-deepseek-ai/DeepSeek-V4-Flash}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TP_SIZE=${TP_SIZE:-4}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
export SGLANG_DSV4_COMPRESS_STATE_DTYPE=${SGLANG_DSV4_COMPRESS_STATE_DTYPE:-bf16}
export SERVER_ARGS_EXTRA=${SERVER_ARGS_EXTRA:---moe-runner-backend flashinfer_mxfp4 --enable-deepseek-v4-fp4-indexer --disable-flashinfer-autotune}

exec "${ROOT}/scripts/run_kernel_interface_capture.sh"
