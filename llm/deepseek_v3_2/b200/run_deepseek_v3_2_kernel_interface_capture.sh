#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}

export ROOT
export MODEL_SLUG=${MODEL_SLUG:-deepseek_v3_2}
export MODEL=${MODEL:-nvidia/DeepSeek-V3.2-NVFP4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export TP_SIZE=${TP_SIZE:-4}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
export SERVER_ARGS_EXTRA=${SERVER_ARGS_EXTRA:---quantization modelopt_fp4 --moe-runner-backend flashinfer_cutlass --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv --kv-cache-dtype fp8_e4m3 --tool-call-parser deepseekv32 --reasoning-parser deepseek-v3 --enforce-disable-flashinfer-allreduce-fusion --disable-flashinfer-autotune}

exec "${ROOT}/scripts/run_kernel_interface_capture.sh"
