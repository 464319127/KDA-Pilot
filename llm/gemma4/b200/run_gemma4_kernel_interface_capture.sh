#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}

export ROOT
export MODEL_SLUG=${MODEL_SLUG:-gemma4}
export MODEL=${MODEL:-google/gemma-4-26B-A4B-it}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}
export TP_SIZE=${TP_SIZE:-1}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
export SERVER_ARGS_EXTRA=${SERVER_ARGS_EXTRA:---reasoning-parser gemma4 --tool-call-parser gemma4 --attention-backend triton --disable-flashinfer-autotune}

exec "${ROOT}/scripts/run_kernel_interface_capture.sh"
