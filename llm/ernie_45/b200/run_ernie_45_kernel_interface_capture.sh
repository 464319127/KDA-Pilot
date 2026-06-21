#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}

export ROOT
export MODEL_SLUG=${MODEL_SLUG:-ernie_45}
export MODEL=${MODEL:-baidu/ERNIE-4.5-21B-A3B-PT}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TP_SIZE=${TP_SIZE:-1}
export MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.75}
export DISABLE_CUDA_GRAPH=${DISABLE_CUDA_GRAPH:-0}
export SERVER_ARGS_EXTRA=${SERVER_ARGS_EXTRA:---disable-cuda-graph --disable-piecewise-cuda-graph}

exec "${ROOT}/scripts/run_kernel_interface_capture.sh"
