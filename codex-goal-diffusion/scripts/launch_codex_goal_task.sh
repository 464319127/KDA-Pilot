#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/launch_codex_goal_task.sh <task-slug-or-dir> [extra codex args...]

Launch a Codex Goal run from codex-goal-diffusion/<task>/ so all generated
artifacts stay inside the Codex Goal task folder.

The launcher prepares:
  <task>/my/plan.md    Copy of <task>/plan.md used by the default prompt
  <task>/baseline/
  <task>/solution/
  <task>/bench/
  <task>/docs/

Default prompt:
  /goal follow the instruction in my/plan.md

Environment overrides:
  CODEX_BIN             Codex executable (default: codex)
  CODEX_MODEL           Optional model passed as --model
  CODEX_PROFILE         Optional profile passed as --profile
  KDA_CODEX_PROMPT      Override the default prompt
  KDA_CODEX_SEARCH=1    Add --search
  KDA_CODEX_BYPASS=1    Add --dangerously-bypass-approvals-and-sandbox
  KDA_CODEX_SANDBOX     Optional sandbox value passed as --sandbox
  KDA_CODEX_APPROVAL_NEVER=1
                        Add --ask-for-approval never
  KDA_REFRESH_PLAN=1    Overwrite my/plan.md when it differs from plan.md
  KDA_BOOTSTRAP_TASK_DIRS=0
                        Do not pre-create baseline/solution/bench/docs
  KDA_NO_CODEX=1        Prepare the task and print the codex command only
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
MY_DIR="$TASK_DIR/my"
MY_PLAN="$MY_DIR/plan.md"

if [[ ! -f "$PLAN_FILE" ]]; then
  echo "error: missing plan.md in task directory: $PLAN_FILE" >&2
  exit 2
fi

mkdir -p "$MY_DIR"
PLAN_STATUS="already current"
if [[ ! -f "$MY_PLAN" ]]; then
  cp "$PLAN_FILE" "$MY_PLAN"
  PLAN_STATUS="created"
elif cmp -s "$PLAN_FILE" "$MY_PLAN"; then
  PLAN_STATUS="already current"
elif is_truthy "${KDA_REFRESH_PLAN:-0}"; then
  cp "$PLAN_FILE" "$MY_PLAN"
  PLAN_STATUS="refreshed"
else
  PLAN_STATUS="kept existing copy; differs from plan.md (set KDA_REFRESH_PLAN=1 to overwrite)"
fi

if ! is_truthy "${KDA_BOOTSTRAP_TASK_DIRS:-1}"; then
  TASK_DIR_STATUS="skipped"
else
  mkdir -p "$TASK_DIR/baseline" "$TASK_DIR/solution" "$TASK_DIR/bench" "$TASK_DIR/docs"
  TASK_DIR_STATUS="ready"
fi

CODEX_BIN="${CODEX_BIN:-codex}"
PROMPT="${KDA_CODEX_PROMPT:-/goal follow the instruction in my/plan.md}"
CODEX_ARGS=(--cd "$TASK_DIR")

if [[ -n "${CODEX_MODEL:-}" ]]; then
  CODEX_ARGS+=(--model "$CODEX_MODEL")
fi
if [[ -n "${CODEX_PROFILE:-}" ]]; then
  CODEX_ARGS+=(--profile "$CODEX_PROFILE")
fi
if is_truthy "${KDA_CODEX_SEARCH:-0}"; then
  CODEX_ARGS+=(--search)
fi
if [[ -n "${KDA_CODEX_SANDBOX:-}" ]]; then
  CODEX_ARGS+=(--sandbox "$KDA_CODEX_SANDBOX")
fi
if is_truthy "${KDA_CODEX_APPROVAL_NEVER:-0}"; then
  CODEX_ARGS+=(--ask-for-approval never)
fi
if is_truthy "${KDA_CODEX_BYPASS:-0}"; then
  CODEX_ARGS+=(--dangerously-bypass-approvals-and-sandbox)
fi
if [[ $# -gt 0 ]]; then
  CODEX_ARGS+=("$@")
fi

echo "== KDA-Pilot Codex Goal task =="
echo "goal root: $GOAL_ROOT"
echo "task:      $TASK_SLUG"
echo "cwd:       $TASK_DIR"
echo "plan:      $MY_PLAN ($PLAN_STATUS)"
echo "outputs:   $TASK_DIR/{baseline,solution,bench,docs} ($TASK_DIR_STATUS)"
echo "prompt:    $PROMPT"
echo

if is_truthy "${KDA_NO_CODEX:-0}"; then
  echo "KDA_NO_CODEX=1 set; prepared task without launching Codex."
  echo
  shell_quote_command "$CODEX_BIN" "${CODEX_ARGS[@]}" "$PROMPT"
  exit 0
fi

if ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
  echo "error: Codex executable not found: $CODEX_BIN" >&2
  exit 127
fi

exec "$CODEX_BIN" "${CODEX_ARGS[@]}" "$PROMPT"
