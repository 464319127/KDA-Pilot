#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KDA_LAUNCHER_NAME="${KDA_LAUNCHER_NAME:-$(basename "$0")}"
export KDA_LAUNCHER_NAME
# Round-robin GPU assignment across the 8 B200 cards (override by exporting KDA_GPU_ID).
KDA_GPU_ID="${KDA_GPU_ID:-3}"
export KDA_GPU_ID
exec "$SCRIPT_DIR/../launch_kda_kernel_task.sh" "llm/nemotron3_ultra/b200/kernels/sgl_kernel_moe_align_block_size" "$@"
