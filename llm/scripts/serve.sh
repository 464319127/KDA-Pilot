#!/usr/bin/env bash
set -euo pipefail
# Start an SGLang server in the background and block until it is ready.
#
# Usage:
#   PORT=30000 SGLANG_TORCH_PROFILER_DIR=/path/profile \
#     llm/scripts/serve.sh <log_file> <ready_timeout_sec> -- <sglang serve cmd...>
#
# Example:
#   llm/scripts/serve.sh /tmp/mm27.log 2400 -- \
#     sglang serve --model-path MiniMaxAI/MiniMax-M2.7 --tp 8 --ep 8 \
#       --tool-call-parser minimax-m2 --reasoning-parser minimax-append-think \
#       --trust-remote-code --mem-fraction-static 0.85 --host 0.0.0.0 --port 30000

LOG="${1:?log file required}"; shift
TIMEOUT="${1:-1800}"; shift || true
[[ "${1:-}" == "--" ]] && shift
[[ $# -ge 1 ]] || { echo "error: no serve command after --" >&2; exit 2; }

PORT="${PORT:-30000}"
echo "[serve] log=$LOG timeout=${TIMEOUT}s port=$PORT"
echo "[serve] cmd: $*"

nohup "$@" >"$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "${LOG}.pid"
echo "[serve] pid=$SERVER_PID (saved to ${LOG}.pid)"

deadline=$(( $(date +%s) + TIMEOUT ))
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[serve] ERROR: server exited early. Tail:"; tail -n 50 "$LOG"; exit 1
  fi
  if grep -q "The server is fired up and ready to roll" "$LOG" 2>/dev/null; then
    echo "[serve] READY (log marker)"; break
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/health_generate" >/dev/null 2>&1; then
    echo "[serve] READY (health_generate)"; break
  fi
  if [[ $(date +%s) -ge $deadline ]]; then
    echo "[serve] ERROR: not ready after ${TIMEOUT}s. Tail:"; tail -n 50 "$LOG"; exit 1
  fi
  sleep 5
done
echo "[serve] server up on port $PORT (pid $SERVER_PID)"
