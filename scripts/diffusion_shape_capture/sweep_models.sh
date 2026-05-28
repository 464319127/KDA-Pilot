#!/usr/bin/env bash
# Run sequential shape-capture passes for a list of diffusion presets.
#
# Required env from caller:
#   HOST_LABEL       (e.g., ion-b200)
#   ARCH_LABEL       (e.g., b200 or h200)
#   GPU_LIST_1GPU    (e.g., "0,1,2,3,4,5,6,7")  - GPUs usable for 1-GPU presets
#   GPU_LIST_4GPU    (e.g., "0,1,2,3")           - GPUs for 4-GPU presets
#   HF_TOKEN
#
# Args:
#   $1: comma-separated list of preset slugs
#   $2: output JSONL log path

set -uo pipefail

presets_csv="$1"
log_path="$2"

CAP_DIR="${CAP_DIR:-/root/diffusion_shape_capture}"
BENCH_PY="${BENCH_PY:-/root/bench_diffusion_denoise.py}"
SGLANG_DIR="${SGLANG_DIR:-/home/sglang-omni/bbuf/repos/sglang}"

export PYTHONPATH="${CAP_DIR}:${SGLANG_DIR}/python:${PYTHONPATH:-}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export DIFFUSION_SHAPE_LOG="$log_path"
export DIFFUSION_SHAPE_HOST="${HOST_LABEL:-unknown}"
export DIFFUSION_SHAPE_ARCH="${ARCH_LABEL:-unknown}"

mkdir -p "$(dirname "$log_path")" /tmp/diff_cap_out

cd "$SGLANG_DIR"
IFS=',' read -ra presets <<< "$presets_csv"

GPU_LIST_1GPU_VAL="${GPU_LIST_1GPU:-0}"
GPU_LIST_4GPU_VAL="${GPU_LIST_4GPU:-0,1,2,3}"

needs_4gpu() {
  case "$1" in
    wan-t2v|wan-i2v|mova-720p) return 0 ;;
    *) return 1 ;;
  esac
}

for preset in "${presets[@]}"; do
  export DIFFUSION_SHAPE_MODEL="$preset"
  if needs_4gpu "$preset"; then
    export CUDA_VISIBLE_DEVICES="$GPU_LIST_4GPU_VAL"
  else
    first_gpu="${GPU_LIST_1GPU_VAL%%,*}"
    export CUDA_VISIBLE_DEVICES="$first_gpu"
  fi
  echo "============================================================"
  echo "[sweep] preset=$preset host=${HOST_LABEL} arch=${ARCH_LABEL} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES log=$log_path"
  echo "============================================================"
  # Use a per-preset timeout so a stuck model does not block the sweep.
  timeout --signal=KILL --kill-after=60 1500 python3 "$BENCH_PY" \
    --model "$preset" \
    --label cap \
    --output-dir /tmp/diff_cap_out \
    --no-torch-compile 2>&1 | tail -150
  rc=${PIPESTATUS[0]}
  echo "[sweep] preset=$preset exit=$rc"
done

echo "[sweep] DONE host=${HOST_LABEL} arch=${ARCH_LABEL} log=$log_path"
echo "[sweep] log lines: $(wc -l < "$log_path" 2>/dev/null || echo 0)"
