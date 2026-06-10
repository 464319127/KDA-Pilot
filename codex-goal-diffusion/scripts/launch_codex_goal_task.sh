#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/launch_codex_goal_task.sh <task-slug-or-dir> [extra codex args...]

Launch a Codex Goal run from codex-goal-diffusion/<task>/ so all generated
artifacts stay inside the Codex Goal task folder.

The launcher prepares:
  <task>/baseline/
  <task>/solution/
  <task>/bench/
  <task>/docs/

Default prompt:
  /goal follow the instruction in plan.md

Environment overrides:
  CODEX_BIN             Codex executable (default: codex)
  CODEX_MODEL           Optional model passed as --model
  CODEX_PROFILE         Optional profile passed as --profile
  CODEX_GOAL_PROMPT     Override the default prompt
  CODEX_GOAL_SEARCH=1   Add --search
  CODEX_GOAL_BYPASS=1   Add --dangerously-bypass-approvals-and-sandbox
  CODEX_GOAL_SANDBOX    Optional sandbox value passed as --sandbox
  CODEX_GOAL_APPROVAL_NEVER=1
                        Add --ask-for-approval never
  CODEX_GOAL_BOOTSTRAP_TASK_DIRS=0
                        Do not pre-create baseline/solution/bench/docs
  CODEX_GOAL_NO_CODEX=1 Prepare the task and print the codex command only
EOF
}

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
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

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

TASK_ARG="${1%/}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
GOAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
PWD_PHYSICAL="$(pwd -P)"

resolve_task_dir() {
  local arg="$1"
  local candidate

  if [[ "$arg" = /* ]]; then
    candidate="$arg"
  elif [[ -d "$PWD_PHYSICAL/$arg" ]]; then
    candidate="$PWD_PHYSICAL/$arg"
  elif [[ "$arg" == codex-goal-diffusion/* && -d "$GOAL_ROOT/${arg#codex-goal-diffusion/}" ]]; then
    candidate="$GOAL_ROOT/${arg#codex-goal-diffusion/}"
  else
    candidate="$GOAL_ROOT/$arg"
  fi

  if [[ ! -d "$candidate" ]]; then
    echo "error: task directory does not exist: $arg" >&2
    exit 2
  fi

  (cd "$candidate" && pwd -P)
}

TASK_DIR="$(resolve_task_dir "$TASK_ARG")"
case "$TASK_DIR" in
  "$GOAL_ROOT"/*) ;;
  *)
    echo "error: task must live under $GOAL_ROOT: $TASK_DIR" >&2
    exit 2
    ;;
esac

TASK_SLUG="${TASK_DIR##*/}"
PLAN_FILE="$TASK_DIR/plan.md"

if [[ ! -f "$PLAN_FILE" ]]; then
  echo "error: missing plan.md in task directory: $PLAN_FILE" >&2
  exit 2
fi

PLAN_STATUS="ready"

if ! is_truthy "${CODEX_GOAL_BOOTSTRAP_TASK_DIRS:-1}"; then
  TASK_DIR_STATUS="skipped"
else
  mkdir -p "$TASK_DIR/baseline" "$TASK_DIR/solution" "$TASK_DIR/bench" "$TASK_DIR/docs"
  TASK_DIR_STATUS="ready"
fi

CODEX_BIN="${CODEX_BIN:-codex}"
PROMPT="${CODEX_GOAL_PROMPT:-/goal follow the instruction in plan.md}"
CODEX_ARGS=(--cd "$TASK_DIR")

if [[ -n "${CODEX_MODEL:-}" ]]; then
  CODEX_ARGS+=(--model "$CODEX_MODEL")
fi
if [[ -n "${CODEX_PROFILE:-}" ]]; then
  CODEX_ARGS+=(--profile "$CODEX_PROFILE")
fi
if is_truthy "${CODEX_GOAL_SEARCH:-0}"; then
  CODEX_ARGS+=(--search)
fi
if [[ -n "${CODEX_GOAL_SANDBOX:-}" ]]; then
  CODEX_ARGS+=(--sandbox "$CODEX_GOAL_SANDBOX")
fi
if is_truthy "${CODEX_GOAL_APPROVAL_NEVER:-0}"; then
  CODEX_ARGS+=(--ask-for-approval never)
fi
if is_truthy "${CODEX_GOAL_BYPASS:-0}"; then
  CODEX_ARGS+=(--dangerously-bypass-approvals-and-sandbox)
fi
if [[ $# -gt 0 ]]; then
  CODEX_ARGS+=("$@")
fi

echo "== Codex Goal task =="
echo "goal root: $GOAL_ROOT"
echo "task:      $TASK_SLUG"
echo "cwd:       $TASK_DIR"
echo "plan:      $PLAN_FILE ($PLAN_STATUS)"
echo "outputs:   $TASK_DIR/{baseline,solution,bench,docs} ($TASK_DIR_STATUS)"
echo "prompt:    $PROMPT"
echo

if is_truthy "${CODEX_GOAL_NO_CODEX:-0}"; then
  echo "CODEX_GOAL_NO_CODEX=1 set; prepared task without launching Codex."
  echo
  shell_quote_command "$CODEX_BIN" "${CODEX_ARGS[@]}" "$PROMPT"
  exit 0
fi

if ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
  echo "error: Codex executable not found: $CODEX_BIN" >&2
  exit 127
fi

exec "$CODEX_BIN" "${CODEX_ARGS[@]}" "$PROMPT"
