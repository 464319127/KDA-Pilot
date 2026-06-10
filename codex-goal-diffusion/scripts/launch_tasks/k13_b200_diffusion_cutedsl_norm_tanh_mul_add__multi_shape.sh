#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$SCRIPT_DIR/../launch_codex_goal_task.sh" "b200_diffusion_cutedsl_norm_tanh_mul_add__multi_shape" "$@"
