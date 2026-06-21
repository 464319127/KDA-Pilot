#!/usr/bin/env bash
# SparseSelectMlaDsv32 — GPU0 DSA-indexer self-MLA over selected KV. 7.6% of decode CUDA.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KDA_LAUNCHER_NAME="${KDA_LAUNCHER_NAME:-$(basename "$0")}"
KDA_TASK_LABEL="${KDA_TASK_LABEL:-b200_tilert_sparse_select_mla}"
KDA_REMOTE_GPU="${KDA_REMOTE_GPU:-2}"   # pin benchmarks to GPU2 (3 tasks concurrent on distinct GPUs)
export KDA_LAUNCHER_NAME KDA_TASK_LABEL KDA_REMOTE_GPU
exec "$SCRIPT_DIR/../launch_kda_kernel_task.sh" "TileRT/kernels/b200_tilert_sparse_select_mla(8%)" "$@"
