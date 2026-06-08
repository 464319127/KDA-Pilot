#!/usr/bin/env bash
set -euo pipefail
# Sweep bench_serving over low / mid / high concurrency against a running server.
#
# Usage:
#   PORT=30000 llm/scripts/bench.sh <model> <outdir> [extra dataset args...]
#
# Defaults to the cookbook "random 1000/1000" dataset method. Override the
# dataset by passing args, e.g. --dataset-name random --random-input-len 1024 ...
#
# Concurrency levels (low/mid/high) and num-prompts are tuned to be stable while
# bounded; anchor low/high to the model's cookbook Speed Benchmark values.

MODEL="${1:?model required}"; OUTDIR="${2:?outdir required}"; shift 2
PORT="${PORT:-30000}"; HOST="${HOST:-127.0.0.1}"

DATASET_ARGS=("$@")
if [[ ${#DATASET_ARGS[@]} -eq 0 ]]; then
  DATASET_ARGS=(--dataset-name random --random-input-len 1000 --random-output-len 1000)
fi

# name:concurrency:num_prompts  (override via LEVELS env, space-separated)
read -r -a LEVELS <<<"${LEVELS:-low:1:10 mid:32:300 high:100:500}"

mkdir -p "$OUTDIR"
for spec in "${LEVELS[@]}"; do
  IFS=: read -r name conc np <<<"$spec"
  log="$OUTDIR/bench_${name}_c${conc}.log"
  echo "[bench] level=$name concurrency=$conc num_prompts=$np -> $log"
  python3 -m sglang.bench_serving \
    --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL" \
    "${DATASET_ARGS[@]}" \
    --num-prompts "$np" --max-concurrency "$conc" 2>&1 | tee "$log"
  echo
done
echo "[bench] done. logs in $OUTDIR"
