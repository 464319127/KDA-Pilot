#!/usr/bin/env bash
# PureMlaDsv32 — MLA decode attention. #1 decode cost (52.8% of TileRT decode CUDA).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KDA_LAUNCHER_NAME="${KDA_LAUNCHER_NAME:-$(basename "$0")}"
KDA_TASK_LABEL="${KDA_TASK_LABEL:-b200_tilert_mla_decode}"
export KDA_LAUNCHER_NAME KDA_TASK_LABEL
exec "$SCRIPT_DIR/../launch_kda_kernel_task.sh" "TileRT/kernels/b200_tilert_mla_decode(53%)" "$@"
