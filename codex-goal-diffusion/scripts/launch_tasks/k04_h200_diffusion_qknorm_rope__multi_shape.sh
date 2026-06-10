#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$SCRIPT_DIR/../launch_codex_goal_task.sh" "h200_diffusion_qknorm_rope__multi_shape" "$@"
