#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$SCRIPT_DIR/../launch_codex_goal_task.sh" "b200_fa4_mha__bf16_head128_total32768" "$@"
