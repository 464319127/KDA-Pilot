#!/usr/bin/env bash
set -euo pipefail
# Capture a torch profiler trace of the serving forward pass under load.
#
# The server MUST have been started with SGLANG_TORCH_PROFILER_DIR set to
# <profile_dir>; the trace files are written there by the server process.
#
# Usage:
#   PORT=30000 llm/scripts/profile_forward.sh <model> <profile_dir> [concurrency] [num_prompts]

MODEL="${1:?model required}"; PDIR="${2:?profile dir required}"
CONC="${3:-32}"; NP="${4:-64}"
PORT="${PORT:-30000}"; HOST="${HOST:-127.0.0.1}"

mkdir -p "$PDIR"
before=$(ls -1 "$PDIR" 2>/dev/null | wc -l | tr -d ' ')

echo "[profile] POST /start_profile"
# Newer SGLang accepts a JSON body; older accepts an empty POST. Try body first.
curl -fsS -X POST "http://${HOST}:${PORT}/start_profile" \
  -H 'Content-Type: application/json' \
  -d "{\"activities\":[\"CPU\",\"GPU\"]}" >/dev/null 2>&1 \
  || curl -fsS -X POST "http://${HOST}:${PORT}/start_profile" >/dev/null 2>&1 \
  || { echo "[profile] ERROR: /start_profile failed"; exit 1; }

echo "[profile] load: concurrency=$CONC num_prompts=$NP (random 1000/1000)"
python3 -m sglang.bench_serving \
  --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL" \
  --dataset-name random --random-input-len 1000 --random-output-len 1000 \
  --num-prompts "$NP" --max-concurrency "$CONC" >"$PDIR/profile_load.log" 2>&1 || true

echo "[profile] POST /stop_profile"
curl -fsS -X POST "http://${HOST}:${PORT}/stop_profile" >/dev/null 2>&1 || true

# The server flushes traces asynchronously; wait for new files to appear.
for _ in $(seq 1 60); do
  now=$(ls -1 "$PDIR" 2>/dev/null | wc -l | tr -d ' ')
  [[ "$now" -gt "$before" ]] && break
  sleep 2
done
echo "[profile] trace dir contents:"
ls -lh "$PDIR" || true
echo "[profile] newest trace:"
ls -1t "$PDIR"/*.json* 2>/dev/null | head -1 || echo "  (none found - check SGLANG_TORCH_PROFILER_DIR matched $PDIR)"
