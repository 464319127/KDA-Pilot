# KDA Kernel Export Tooling

Promote a finished KDA kernel task into the shippable `kda_kernels/`
package, and verify the resulting overlay.

## Promote one task

```bash
cd /Users/bbuf/工作目录/Common/kernel-pilot
python3 scripts/export_kda_kernels/export.py b200_diffusion_qknorm_rope__multi_shape
```

This copies the task's `src/` into
`kda_kernels/diffusion/qknorm_rope/_impls/b200/`, rewires the matching
kda_kernels family package (`kda_kernels/diffusion/qknorm_rope/`) to import
from the generated architecture dispatcher, and flips
`KDA_OPTIMIZED_<fn> = True`. The corresponding `KDA_ARCHES_<fn>`,
`KDA_TASK_<fn>`, `KDA_COMMIT_<fn>`, `KDA_DATE_<fn>`, and
`KDA_SPEEDUP_<fn>` stamps are recorded.

Export the H200 mirror when it is ready:

```bash
python3 scripts/export_kda_kernels/export.py h200_diffusion_qknorm_rope__multi_shape
```

Both arch implementations remain under the same family. At runtime the
dispatcher selects B200 for CUDA capability `(10, 0)`, H200 for `(9, 0)`, and
falls back to the captured SGLang baseline when no promoted implementation
matches the current device or call signature.

## List all tasks and their export state

```bash
python3 scripts/export_kda_kernels/export.py --list
```

Shows `[exported]` next to tasks that already have an impl copy for that task's
own arch. A B200 export does not mark the H200 mirror exported.

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

Removes only that arch's impl directory and rewrites the affected
kda_kernels family. If another arch is still exported for the same function,
`KDA_OPTIMIZED_<fn>` stays `True` and the dispatcher keeps serving that arch;
otherwise the function falls back to the baseline-re-export shape with
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
