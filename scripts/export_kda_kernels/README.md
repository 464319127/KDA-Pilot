# KDA Kernel Export Tooling

Promote a finished KDA kernel task into the shippable `kda_kernels/`
package, and verify the resulting overlay.

## Promote one task

```bash
cd /Users/bbuf/工作目录/Common/kernel-pilot
python3 scripts/export_kda_kernels/export.py b200_diffusion_qknorm_rope__multi_shape
```

This copies the task's `src/` into
`kda_kernels/_impls/b200_diffusion_qknorm_rope__multi_shape/`, rewires
the matching kda_kernels stub
(`kda_kernels/diffusion/qknorm_rope.py`) to import from that impl,
and flips `KDA_OPTIMIZED_<fn> = True`. The corresponding
`KDA_TASK_<fn>`, `KDA_COMMIT_<fn>`, `KDA_DATE_<fn>`, and
`KDA_SPEEDUP_<fn>` stamps are recorded.

## List all tasks and their export state

```bash
python3 scripts/export_kda_kernels/export.py --list
```

Shows `[exported]` next to tasks that already have an impl copy.

## Verify the package + the patch

```bash
export PYTHONPATH=/Users/bbuf/工作目录/Common/kernel-pilot:$PYTHONPATH
python3 scripts/export_kda_kernels/verify.py
```

Prints a per-function status table and exits non-zero if any registry
entry skipped due to an exception (as opposed to the expected
"skipped: not optimized" when nothing has been exported yet).

## Revert one task

```bash
python3 scripts/export_kda_kernels/export.py --revert b200_diffusion_qknorm_rope__multi_shape
```

Removes the impl directory and rewrites the affected kda_kernels
stub back to the baseline-re-export shape with
`KDA_OPTIMIZED_<fn> = False`.

## Task `src/register.py` contract

For the export to know which functions to swap, the task's
`src/register.py` should expose an `EXPORTS` dict keyed by the
target sglang function name:

```python
# kernels/<task>/src/register.py

import torch

def fused_inplace_qknorm_rope(q, k, q_weight, k_weight,
                              cos_sin_cache, positions, *, is_neox,
                              eps=1e-6, head_dim=0, rope_dim=0):
    # ... KDA-optimized impl ...
    return None

# Required for the export tool to wire this into kda_kernels:
EXPORTS = {
    "fused_inplace_qknorm_rope": fused_inplace_qknorm_rope,
}

# Existing register() / optimized_wrapper() entries continue to live
# here for the local benchmark.py / test_correctness.py harness.
```

Multi-function task families (`fuse_scale_shift`, `norm_infer`,
`cutedsl_norm_*`, `group_norm_silu`, `rotary_embedding`) list every
promoted function under the same `EXPORTS` dict; the export tool
flips only the listed flags and leaves the rest on baseline.
