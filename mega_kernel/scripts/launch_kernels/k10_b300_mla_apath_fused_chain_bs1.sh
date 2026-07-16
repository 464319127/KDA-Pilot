#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KDA_LAUNCHER_NAME="${KDA_LAUNCHER_NAME:-$(basename "$0")}"
export KDA_LAUNCHER_NAME
CLAUDE_MODEL="${CLAUDE_MODEL:-fable}"
export CLAUDE_MODEL
exec "$SCRIPT_DIR/../launch_kda_kernel_task.sh" "mega_kernel/tasks/10" "$@"
