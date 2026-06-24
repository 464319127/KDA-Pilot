#!/usr/bin/env bash
# Fetch the exact CUTLASS commit that sgl-kernel FetchContent's at
# sglang main@34dd9c28 (see docs/baseline_source.md) into <task>/.deps/cutlass.
# CUTLASS is a build-time dependency only; it is git-ignored and never staged in
# the PR. Uses the launcher-selected modern bash via #!/usr/bin/env bash.
set -euo pipefail

PIN="57e3cfb47a2d9e0d46eb6335c3dc411498efa198"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEST="${CUTLASS_DIR:-$ROOT/.deps/cutlass}"

if [[ -f "$DEST/include/cutlass/cutlass.h" ]]; then
  echo "CUTLASS already present at $DEST"
  exit 0
fi

mkdir -p "$DEST"
cd "$DEST"
git init -q
git remote add origin https://github.com/NVIDIA/cutlass.git 2>/dev/null || true
echo "Fetching NVIDIA/cutlass@$PIN (shallow) ..."
git fetch -q --depth 1 origin "$PIN"
git checkout -q FETCH_HEAD
echo "CUTLASS @ $(git rev-parse HEAD) checked out at $DEST"
test -f "$DEST/include/cutlass/cutlass.h" && echo "OK: cutlass.h present"
