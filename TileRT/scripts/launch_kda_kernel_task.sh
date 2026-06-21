#!/usr/bin/env bash
set -euo pipefail

# One-click KDA launcher for TileRT DeepSeek-V3.2 kernel-optimization tasks.
# Mirrors diffusion/scripts/launch_kda_kernel_task.sh, but bootstraps a
# TileRT-specific gen-plan draft (golden_forward oracle + ≥3× ncu reference +
# TileRT design levers) instead of the diffusion (upstream-SGLang-copy) flow.

usage() {
  cat <<'EOF'
Usage:
  TileRT/scripts/launch_kda_kernel_task.sh <kernel-task-dir> [extra claude args...]

Creates a task-owned git worktree, enters the TileRT kernel task directory inside
that worktree, bootstraps a Humanize gen-plan draft tuned for TileRT kernels, and
launches Claude Code with CLAUDE_PROJECT_DIR set to the kernel directory so the
official Humanize hooks stay local to that task's .humanize state.

<kernel-task-dir> is repo-relative, e.g.
  'TileRT/kernels/b200_tilert_mla_decode(53%)'

Environment overrides (same contract as the diffusion launcher):
  KDA_BASE_BRANCH       Base branch/ref for the worktree
                        (default: current checkout branch, or HEAD if detached)
  KDA_WORKTREE_BASE     Parent dir for generated worktrees
                        (default: ../KDA-Pilot-worktrees next to this repo)
  KDA_RUN_ID            Run suffix (default: timestamp-pid)
  KDA_BRANCH            Exact branch name to create
  KDA_BRANCH_PREFIX     Branch prefix when KDA_BRANCH is unset (default: kda)
  KDA_REVIEW_BASE       Exact local branch name for the RLCR review base
  KDA_REVIEW_BASE_PREFIX  Review-base prefix when unset (default: kda-base)
  KDA_WORKTREE_ROOT     Exact worktree path to create
  KDA_TASK_LABEL        Friendly label for branch/worktree names (per-kernel
                        wrappers set this to a clean slug so the '(NN%)' folder
                        suffix never leaks into git branch/worktree names)
  CLAUDE_BIN            Claude executable (default: claude)
  CLAUDE_MODEL          Claude model flag value (default: opus)
  CLAUDE_EFFORT         Claude effort flag value (default: max)
  KDA_BASH_BIN          Modern bash for launch + spawned Claude hooks
  KDA_LAUNCHER_NAME     Friendly launcher/task-card name (set by wrappers)
  KDA_BOOTSTRAP_DRAFT=0 Skip automatic .humanize/kernel-agent/draft.md creation
  KDA_NO_CLAUDE=1       Create the worktree and print commands without launching
  IS_SANDBOX            Forwarded to Claude (default: 1)
  HUMANIZE_CODEX_BYPASS_SANDBOX  Forwarded to Claude (default: true)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

TASK_DIR="${1%/}"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
if [[ -n "${KDA_BASE_BRANCH:-}" ]]; then
  BASE_BRANCH="$KDA_BASE_BRANCH"
else
  BASE_BRANCH="$(
    git -C "$REPO_ROOT" symbolic-ref --quiet --short HEAD ||
      git -C "$REPO_ROOT" rev-parse --verify HEAD
  )"
fi
DEFAULT_WORKTREE_BASE="$(cd "$REPO_ROOT/.." && pwd)/KDA-Pilot-worktrees"
WORKTREE_BASE="${KDA_WORKTREE_BASE:-$DEFAULT_WORKTREE_BASE}"
RUN_ID="${KDA_RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
TASK_SLUG="${TASK_DIR##*/}"
LAUNCHER_NAME="${KDA_LAUNCHER_NAME:-direct}"
TASK_LABEL="${KDA_TASK_LABEL:-${LAUNCHER_NAME%.sh}}"
if [[ "$TASK_LABEL" == "direct" || -z "$TASK_LABEL" ]]; then
  # Fall back to the folder slug, but strip the '(NN%)' decode-share suffix and
  # any other chars that are awkward in git branch / worktree names.
  TASK_LABEL="$(printf '%s' "$TASK_SLUG" | sed -E 's/\([0-9]+%\)$//; s/[^A-Za-z0-9._-]+/_/g')"
fi
BRANCH_PREFIX="${KDA_BRANCH_PREFIX:-kda}"
BRANCH="${KDA_BRANCH:-${BRANCH_PREFIX}/${TASK_LABEL}-${RUN_ID}}"
REVIEW_BASE_PREFIX="${KDA_REVIEW_BASE_PREFIX:-kda-base}"
REVIEW_BASE="${KDA_REVIEW_BASE:-${REVIEW_BASE_PREFIX}/${TASK_LABEL}-${RUN_ID}}"
WORKTREE_ROOT="${KDA_WORKTREE_ROOT:-${WORKTREE_BASE}/${TASK_LABEL}-${RUN_ID}}"
CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CLAUDE_MODEL="${CLAUDE_MODEL:-opus}"
CLAUDE_EFFORT="${CLAUDE_EFFORT:-max}"

# TileRT tasks are B200-only (sm_100). Default remote host = the cirrascale B200
# node from the project skill; override per task prompt if needed.
TARGET_GPU_LABEL="B200"
REMOTE_HOST_HINT="the cirrascale-gpua83e B200 node (or the host named in the task prompt)"

bash_is_kda_safe() {
  local candidate="$1"
  [[ -n "$candidate" && -x "$candidate" ]] || return 1
  "$candidate" -c 'set -euo pipefail; a=(); : "${a[@]}"; [[ ${BASH_VERSINFO[0]} -gt 3 ]]' >/dev/null 2>&1
}

find_kda_bash() {
  local candidate
  if [[ -n "${KDA_BASH_BIN:-}" ]]; then
    bash_is_kda_safe "$KDA_BASH_BIN" && printf '%s\n' "$KDA_BASH_BIN"
    return
  fi
  for candidate in "$(command -v bash 2>/dev/null || true)" /opt/homebrew/bin/bash /usr/local/bin/bash; do
    if bash_is_kda_safe "$candidate"; then
      printf '%s\n' "$candidate"
      return
    fi
  done
}

KDA_SELECTED_BASH="$(find_kda_bash || true)"
if [[ -z "$KDA_SELECTED_BASH" ]]; then
  echo "error: KDA-Pilot requires a modern bash for Humanize hooks; /bin/bash 3.2 is not supported" >&2
  echo "hint: install Homebrew bash and/or set KDA_BASH_BIN=/opt/homebrew/bin/bash" >&2
  exit 127
fi
KDA_SELECTED_BASH_DIR="$(cd "$(dirname "$KDA_SELECTED_BASH")" && pwd)"
KDA_LAUNCH_PATH="$KDA_SELECTED_BASH_DIR:$PATH"

if ! bash_is_kda_safe "$BASH"; then
  if [[ "${KDA_BASH_REEXECED:-}" == "1" ]]; then
    echo "error: failed to re-exec KDA-Pilot with safe bash: $KDA_SELECTED_BASH" >&2
    exit 127
  fi
  export KDA_BASH_REEXECED=1
  export KDA_BASH_BIN="$KDA_SELECTED_BASH"
  export PATH="$KDA_LAUNCH_PATH"
  exec "$KDA_SELECTED_BASH" "$0" "$TASK_DIR" "$@"
fi

if [[ "$TASK_DIR" = /* || "$TASK_DIR" == *".."* ]]; then
  echo "error: task dir must be repo-relative and must not contain '..': $TASK_DIR" >&2
  exit 2
fi

if [[ ! -d "$REPO_ROOT/$TASK_DIR" ]]; then
  echo "error: task dir does not exist in repo: $REPO_ROOT/$TASK_DIR" >&2
  exit 2
fi

if [[ "${KDA_NO_CLAUDE:-}" != "1" ]] && ! command -v "$CLAUDE_BIN" >/dev/null 2>&1; then
  echo "error: Claude executable not found: $CLAUDE_BIN" >&2
  exit 127
fi

if ! git -C "$REPO_ROOT" rev-parse --verify --quiet "$BASE_BRANCH" >/dev/null; then
  echo "error: base branch/ref not found: $BASE_BRANCH" >&2
  exit 2
fi

if ! git -C "$REPO_ROOT" cat-file -e "$BASE_BRANCH:$TASK_DIR" 2>/dev/null; then
  echo "error: base branch/ref does not contain task dir: $BASE_BRANCH:$TASK_DIR" >&2
  echo "hint: commit the kernel task folder or set KDA_BASE_BRANCH to a ref that contains it" >&2
  exit 2
fi

if [[ -e "$WORKTREE_ROOT" ]]; then
  echo "error: worktree path already exists: $WORKTREE_ROOT" >&2
  echo "hint: set KDA_RUN_ID or KDA_WORKTREE_ROOT to choose a fresh path" >&2
  exit 2
fi

if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$REVIEW_BASE"; then
  echo "error: review base branch already exists: $REVIEW_BASE" >&2
  echo "hint: set KDA_RUN_ID or KDA_REVIEW_BASE to choose a fresh branch" >&2
  exit 2
fi

mkdir -p "$WORKTREE_BASE"

echo "== TileRT KDA task launcher =="
echo "repo:      $REPO_ROOT"
echo "launcher:  $LAUNCHER_NAME"
echo "label:     $TASK_LABEL"
echo "task:      $TASK_DIR"
echo "base:      $BASE_BRANCH"
echo "review:    $REVIEW_BASE"
echo "branch:    $BRANCH"
echo "worktree:  $WORKTREE_ROOT"
echo "bash:      $KDA_SELECTED_BASH ($("$KDA_SELECTED_BASH" --version | head -1))"
echo

git -C "$REPO_ROOT" branch "$REVIEW_BASE" "$BASE_BRANCH"
git -C "$REPO_ROOT" worktree add -b "$BRANCH" "$WORKTREE_ROOT" "$BASE_BRANCH"

cd "$WORKTREE_ROOT/$TASK_DIR"

if [[ "${KDA_BOOTSTRAP_DRAFT:-1}" != "0" ]]; then
  mkdir -p .humanize/kernel-agent
  DRAFT_FILE=".humanize/kernel-agent/draft.md"
  if [[ ! -f "$DRAFT_FILE" ]]; then
    {
      cat <<EOF
# Humanize Gen-Plan Draft For ${TASK_SLUG}

Generate a Humanize RLCR implementation plan to write a B200 CUDA kernel that
**matches TileRT's measured latency** for this DeepSeek-V3.2 fused kernel, while
staying bit-faithful to the PyTorch baseline. Preserve the source prompt's
problem definition, exact shapes/dtypes, baseline contract, correctness oracle,
≥3× ncu reference, shape-coverage set, remote-GPU constraints, and local folder
contract.

## Source Prompt

The task source is the local \`prompt.md\` below.

\`\`\`markdown
EOF
      cat prompt.md
      cat <<EOF
\`\`\`

## Mandatory TileRT/KDA-Pilot Constraints

- Use this current kernel folder as the optimization workspace.
- Keep \`.humanize*\` untracked.
- Use official Humanize commands installed in the agent environment. Do not use
  any vendored/repository-local Humanize implementation from KDA-Pilot.
- **Baseline = the golden reference already in \`baseline/\`** — a faithful port
  of TileRT's own \`golden_forward\` (PyTorch). Do NOT replace it with an
  upstream-SGLang copy; TileRT is closed-source and there is no upstream kernel
  to clone. Expose baseline and candidate through the same low-overhead local
  entry ABI (\`bench/adapter.py\`).
- **Correctness oracle**: validate the candidate against \`baseline/\` via
  \`bench/correctness.py\` over \`bench/workloads.json\`. Where the real op is
  1-GPU isolatable, the authoritative oracle is \`../../harness/tilert_oracle.py\`
  (golden_forward vs the real \`torch.ops.tilert.*\` kernel). Tolerances are in
  \`../../docs/tilert_correctness_contract.md\` (bf16 rel <2e-2, fp8/fp4 <5e-2).
- **Performance target**: the TileRT reference latency in \`config.toml\`
  \`[reference]\` (and \`../../docs/tilert_reference.md\`) = **median of ≥3
  isolated ncu runs** of \`gpu__time_duration.avg\` on an idle ${TARGET_GPU_LABEL}.
  Re-measure with \`../../harness/sweep_ncu.py\`; methodology in
  \`../../docs/benchmark_method.md\`. Beat or match it on every covered shape.
- **Shape coverage**: cover the op's actual AOT-compiled (kSeqLen, KeComputeType)
  set from \`../../KERNEL_REGISTRY.md\` and \`bench/workloads.json\` — decode
  S∈{1,2,4}, plus MTP/prefill seq (3/8/16) and fp8/fp4 CT variants where the
  registry lists them for this op.
- **Design levers** (cite per edit): \`../../docs/tilert_design_levers.md\` —
  persistent grid occupancy=1 (148 CTAs × 384 thr, ~168 reg), warp specialization
  + TMA double-buffer (Prefetcher streams weights GMEM→SMEM, Consumer runs
  warpgroup HMMA, mbarrier handshake), no-GMEM intermediates, native fp8/fp4
  warpgroup MMA, DSA top-2048 sparse KV gather, flag-based NVLink allreduce fused
  into the op.
- Target GPU = ${TARGET_GPU_LABEL} (sm_100). Read
  \`${WORKTREE_ROOT}/external/KernelWiki/SKILL.md\` and
  \`${WORKTREE_ROOT}/external/ncu-report-skill/SKILL.md\` before implementation;
  use KernelWiki for upstream design ideas and ncu-report-skill for
  evidence-backed diagnosis when profiling would change the next edit.
- Do not use macOS \`/bin/bash\` or any Bash 3.x runtime for local Humanize,
  hook, launcher, or helper scripts. The launcher exports \`KDA_BASH_BIN\` and
  prepends its directory to \`PATH\`; preserve that environment.
- Comm ops (allreduce / broadcast / receive): the real op needs NVLink
  peer_bufs/ll_buf + flag exchange, so it is **not isolatable on 1 GPU**. Use the
  reimplemented golden math for correctness and the in-graph profiler time as the
  latency target; do not fake a standalone ncu number.
- Check GPU state before and after every benchmark/profile run; treat numbers as
  valid only when the selected card is idle. Never fabricate benchmark, NCU,
  correctness, topology, or GPU-id evidence.
- Recover K/R/W from the source prompt before implementation:
  - K: kernel semantics + callsite contract
  - R: correctness oracle (golden_forward / \`../../harness/tilert_oracle.py\`) + baseline path
  - W: the (kSeqLen, KeComputeType) workload set + the isolated ≥3× ncu methodology
- In every RLCR iteration, refresh context from the source prompt, KERNEL_REGISTRY
  + design levers, current benchmark/profile evidence, KernelWiki, and
  ncu-report-skill before choosing the next edit, profile, benchmark, or no-go.
- Do not implement kernels, run long benchmarks, or collect NCU evidence before
  RLCR is active.
- Keep candidate code under \`solution/\`, harnesses under \`bench/\`, and
  provenance/results under \`docs/\`. Keep raw NCU/Nsight/build/scratch artifacts
  local for evidence but exclude them from the final PR.
- Always ask before destructive operations, global machine/container changes,
  deleting another task's artifacts, changing shared production checkouts in
  place, or relaxing correctness/baseline/promotion requirements.

## Default Decision Policy

Minimize user-choice prompts during RLCR. Bake in these defaults unless the
source prompt says otherwise:

- Remote ${TARGET_GPU_LABEL} validation is a normal part of the loop. After the
  local scaffold, correctness, and implementation checks are committed, proceed
  to the remote GPU phase autonomously.
- Use the matching remote host (${REMOTE_HOST_HINT}).
- Before GPU work, inspect remote GPU state and pick a ${TARGET_GPU_LABEL} with
  no active compute and no meaningful memory use. Export it as \`REMOTE_GPU_ID\`
  and reuse it for baseline, candidate, benchmark, profiler, and NCU commands.
- Treat minimal, reversible, task-owned setup as approved: remote workspaces,
  checkouts, in-workspace builds, local editable installs, profiler traces,
  benchmark/NCU artifacts.
- If no idle ${TARGET_GPU_LABEL} is available, wait or retry briefly. Ask before
  changing the benchmark environment or running measurements on a busy GPU.
- If the dependency stack cannot run the baseline or load the real \`tilert\` op,
  create a task-owned remote workspace and pin/rebuild deps (tilert wheel + sglang
  container) there. Ask only if the rebuild repeatedly fails or a destructive
  change outside the task workspace would be required.
- Do not finalize an evidence-backed no-go just because the first candidate did
  not win. A no-go needs correctness, baseline numbers, candidate attempts,
  benchmark or NCU evidence, and a named active bound or blocker.

## Expected Plan Shape

- Recover the kernel semantics + callsite contract, the baseline path, and the
  shape/CT set from \`prompt.md\` + \`../../KERNEL_REGISTRY.md\`.
- Define matching baseline and candidate entry points using the same ABI,
  argument order, stream behavior, output allocation, and build path
  (\`bench/adapter.py\`).
- Fill \`bench/correctness.py\` against \`baseline/\` before optimization;
  freeze \`bench/workloads.json\` and the immutable TileRT reference latency.
- Rank candidate directions by expected benefit/risk using the design levers.
- Implement bounded optimization attempts under RLCR; refresh KernelWiki +
  ncu-report-skill context each iteration or note why no new query is needed.
- Use NCU/profile evidence (ncu-report-skill) for non-obvious bottlenecks.
- Include a remote phase recording host/GPU id/model, before/after idleness,
  exact commands, and benchmark/NCU artifacts.
- Update \`docs/results.md\` with per-shape baseline-vs-candidate numbers before
  completion. Before committing or opening a PR, inspect the staged diff and
  exclude raw NCU reports, Nsight traces, profiler directories, build outputs,
  scratch logs, and failed-experiment dumps.
EOF
    } > "$DRAFT_FILE"
  fi
fi

echo
echo "== Claude project root =="
echo "$PWD"
echo
echo "Draft: .humanize/kernel-agent/draft.md"
echo "Inside Claude Code, use:"
echo "/humanize:gen-plan --input .humanize/kernel-agent/draft.md --output .humanize/kernel-agent/refined-plan.md --direct"
echo "/humanize:start-rlcr-loop .humanize/kernel-agent/refined-plan.md --skip-quiz --claude-answer-codex --max 12 --codex-model gpt-5.5:high --codex-timeout 5400 --base-branch $REVIEW_BASE"
echo

if [[ "${KDA_NO_CLAUDE:-}" == "1" ]]; then
  echo "KDA_NO_CLAUDE=1 set; worktree prepared without launching Claude."
  exit 0
fi

exec env \
  PATH="$KDA_LAUNCH_PATH" \
  SHELL="$KDA_SELECTED_BASH" \
  KDA_BASH_BIN="$KDA_SELECTED_BASH" \
  CLAUDE_PROJECT_DIR="$PWD" \
  IS_SANDBOX="${IS_SANDBOX:-1}" \
  HUMANIZE_CODEX_BYPASS_SANDBOX="${HUMANIZE_CODEX_BYPASS_SANDBOX:-true}" \
  "$CLAUDE_BIN" \
  --permission-mode bypassPermissions \
  --model "$CLAUDE_MODEL" \
  --effort "$CLAUDE_EFFORT" \
  "$@"
