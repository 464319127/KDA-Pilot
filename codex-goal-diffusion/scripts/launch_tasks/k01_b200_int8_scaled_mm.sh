#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$SCRIPT_DIR/../launch_codex_goal_task.sh" "b200_int8_scaled_mm__m64_n2048_k2048_bias" "$@"
