#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$SCRIPT_DIR/../launch_codex_goal_task.sh" "b200_diffusion_group_norm_silu__multi_shape" "$@"
