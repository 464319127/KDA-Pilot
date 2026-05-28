#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KDA_LAUNCHER_NAME="${KDA_LAUNCHER_NAME:-$(basename "$0")}"
export KDA_LAUNCHER_NAME
exec "$SCRIPT_DIR/../launch_kda_kernel_task.sh" "kernels/b200_rotary_embedding__diffusion_multi_shape" "$@"
