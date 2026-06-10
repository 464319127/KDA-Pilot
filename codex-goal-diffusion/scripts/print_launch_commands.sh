#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/print_launch_commands.sh [extra codex args...]

Print one launch command per Codex Goal diffusion task. Paste each line into a
separate terminal window when you want to run many tasks in parallel.
EOF
}

shell_quote_command() {
  local arg
  printf '%q' "$1"
  shift
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

for launcher in "$SCRIPT_DIR"/launch_tasks/k*.sh; do
  [[ -f "$launcher" ]] || continue
  shell_quote_command "$launcher" "$@"
done
