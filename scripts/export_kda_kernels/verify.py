#!/usr/bin/env python3
"""Smoke-check kda_kernels: import the package, run install(), and print the
per-function status table. Exits 0 if every entry in the registry either
swapped or skipped cleanly (no exception).
"""

from __future__ import annotations

import sys

try:
    import kda_kernels
except ImportError as e:
    print(f"FAIL: cannot import kda_kernels ({e!r})", file=sys.stderr)
    sys.exit(2)

results = kda_kernels.install()
ok = True
print(f"{'status':<26s}  {'sglang path':<70s}  ->  kda path")
print("-" * 140)
for sglang_path, kda_path, status in results:
    print(f"{status:<26s}  {sglang_path:<70s}  ->  {kda_path}")
    if status.startswith("skipped:") and not (
        status == "skipped: not optimized"
        or status == "skipped: already installed"
    ):
        ok = False

print()
print(f"installed: {len(kda_kernels.status())} swaps")
if not ok:
    print("FAIL: at least one entry skipped due to an exception", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
