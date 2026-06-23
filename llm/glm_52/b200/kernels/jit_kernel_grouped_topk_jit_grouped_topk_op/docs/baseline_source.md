# Baseline Source Provenance — `jit_kernel.grouped_topk._jit_grouped_topk_op`

## Upstream lineage

| Field | Value |
|-------|-------|
| Repository | https://github.com/sgl-project/sglang |
| Branch | `main` |
| Resolved commit | `6b2c730bf793984c39f7f07b3c074ca05b059b00` |
| Resolution method | `git ls-remote https://github.com/sgl-project/sglang.git refs/heads/main` |
| Resolution time (UTC) | 2026-06-22T15:40:15Z |
| Recovery method | `git archive 6b2c730 <paths>` on host `ion-b200`, container `sglang_bbuf`, extracted into `baseline/` (verbatim, no edits) |
| Local edits | None. Files copied byte-for-byte from the pinned commit. |

## Python interface ↔ source mapping

The captured Python interface `jit_kernel.grouped_topk._jit_grouped_topk_op`
resolves to `python/sglang/jit_kernel/grouped_topk.py`:

- `_jit_grouped_topk_op(scores, bias, topk_values, topk_indices, num_expert_group, topk_group, topk, renormalize, scaling_factor) -> None`
  is a `@register_custom_op(mutates_args=["topk_values","topk_indices"])` —
  **destination-passing**: it writes into the caller-provided `topk_values`
  `(N, topk)` f32 and `topk_indices` `(N, topk)` i32 buffers and returns `None`.
- It dispatches to a JIT module compiled from the single CUDA header
  `python/sglang/jit_kernel/csrc/moe/grouped_topk.cuh`, exported through TVM-FFI
  as the typed function `grouped_topk`.
- The public `grouped_topk(...)` wrapper allocates the two output tensors and
  returns the tuple `(topk_values, topk_indices)` — i.e. **exactly two** tensors.

### Capture result-tensor reconciliation

The runtime API capture logged two outputs for this call site: `return=None`
(the destination-passing op itself) and a separate `(N,8)f32, (N,8)i32,
(N,256)bf16` tuple. The recovered source shows `grouped_topk` returns only the
two `(N,8)` tensors, so the `(N,256) bf16` tensor is **not** part of this op's
output contract — it is an artifact of the surrounding MoE routing layer
(`python/sglang/srt/layers/moe/topk.py`), not of `_jit_grouped_topk_op`. The
local baseline and candidate therefore target the destination-passing
`(scores, bias, topk_values_out, topk_indices_out, ...)` ABI.

## Copied files

```
baseline/csrc/moe/grouped_topk.cuh             # the kernel (verbatim)
baseline/include/sgl_kernel/tensor.h           # TensorMatcher, SymbolicSize, SymbolicDevice
baseline/include/sgl_kernel/utils.h            # RuntimeCheck, div_ceil, source_location
baseline/include/sgl_kernel/utils.cuh          # LaunchKernel, fp32_t, device helpers
baseline/include/sgl_kernel/source_location.h  # DebugInfo / panic location
```

These four headers are the exact transitive `sgl_kernel` include closure of
`grouped_topk.cuh` (`tensor.h → utils.h, utils.cuh`; `utils.h → source_location.h`;
`utils.cuh → utils.h`). The full upstream `include/sgl_kernel/` tree was recovered
at commit `6b2c730`, then pruned to this closure — verified sufficient by a clean
build of both baseline and candidate plus the full correctness grid (1479 checks).
External framework headers (`dlpack/*`, `tvm/ffi/*`, `cuda_*`) are **not** copied —
they are provided by the installed `apache-tvm-ffi` package and the CUDA toolkit,
exactly as the upstream JIT does. The candidate (`solution/`) reuses the same four
headers for build parity.

## Recovered kernel semantics (the correctness oracle contract)

`grouped_topk_single_group_kernel<MaxExperts>` (one CUDA block per token,
`MaxExperts` ∈ {128, 256, 512} threads chosen by expert count):

1. **Score**: per expert `e`, `s = sigmoid(scores[e])` with
   `sigmoid(x) = 1 / (1 + __expf(-x))` (single-precision, fast intrinsic — **not**
   softmax), then `biased = s + bias[e]`.
2. **Select**: top-`topk` experts by **biased** score, chosen iteratively in
   descending order via a packed `(value, index)` warp-max reduction.
3. **Tie-break**: equal biased scores resolve to the **smaller expert index**
   (the pack encodes `65535 - idx`, so a smaller index yields a larger packed key).
4. **Weights**: the emitted weight for a selected expert is its **un-biased**
   sigmoid value `s` (not the biased score).
5. **Renormalize** (when `renormalize=True`): divide each selected weight by
   `(sum of the topk selected weights + 1e-20)`.
6. **Scale**: multiply each weight by `scaling_factor`.
7. **Output order**: descending by biased score (selection order), written to
   `topk_values[token, 0..topk-1]` and `topk_indices[token, 0..topk-1]`.

### Supported domain (RuntimeCheck in the launcher)

- `num_expert_group == 1 && topk_group == 1` (single-group only)
- `topk <= 8`
- `num_experts <= 512`
- `num_tokens == 0` is a valid no-op (returns immediately)

All captured GLM-5.2 production calls (17404/17404) fall inside this domain:
`E=256, topk=8, num_expert_group=1, topk_group=1, renormalize=True,
scaling_factor=1.0`, `scores` f32 `(N,256)`, `bias` f32 `(256,)`.

## Build recipe (matches upstream JIT, used for both baseline and candidate)

Header-only TVM-FFI `load_inline` with a generated wrapper
`TVM_FFI_DLL_EXPORT_TYPED_FUNC(grouped_topk, (grouped_topk));`:

- C++ flags: `-std=c++20 -O3`
- CUDA flags (B200 / sm_100): `-DSGL_CUDA_ARCH=1000 -std=c++20 -O3 --expt-relaxed-constexpr`
  (`TVM_FFI_CUDA_ARCH_LIST=10.0`)
- Include paths: `baseline/include` (copied `sgl_kernel/*`) + the `apache-tvm-ffi`
  package include dir (`dlpack/*`, `tvm/ffi/*`) + CUDA toolkit.

The candidate in `solution/` exposes the **same** `grouped_topk(...)`
`tvm::ffi::TensorView` signature, is built with the **same** flags and include
paths, and launches on `at::cuda::getCurrentCUDAStream()` — so the only measured
difference is the kernel body.

## Baseline bottleneck (for candidate direction)

The recovered kernel launches **one block per token with `MaxExperts` threads**
(256 for E=256) but performs the top-k selection in **warp 0 only** — the other
7 warps (224 threads) go idle after the shared-memory staging phase. Selection is
`topk` (8) **sequential** re-scan + warp-reduce passes over shared memory, each
masking the chosen expert to `-FLT_MAX`. For the dominant decode regime (N≤38,
~85% of calls) this both under-occupies each block and launches one block per
token. Candidate headroom is in: warp-per-token (multiple tokens per CTA) to cut
block count and idle threads, register-resident local-top-8-then-single-merge to
replace the 8 sequential shared-memory passes, and avoiding the dual
shared-memory staging.
