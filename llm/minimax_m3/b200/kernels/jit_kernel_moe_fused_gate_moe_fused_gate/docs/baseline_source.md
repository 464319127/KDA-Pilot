# Baseline Source Provenance — `moe_fused_gate`

## Upstream resolution

| Field | Value |
|-------|-------|
| Repository | https://github.com/sgl-project/sglang |
| Branch | `main` |
| Resolved commit | `34dd9c28caf4f7dd185e58e462a1344b52568e2e` |
| Resolution method | `git ls-remote https://github.com/sgl-project/sglang.git refs/heads/main` |
| Resolution time (UTC) | 2026-06-23T23:49:54Z |
| Retrieval method | `git fetch --depth 1 --filter=blob:none <SHA>` + cone sparse-checkout of `python/sglang/jit_kernel/include` and `python/sglang/jit_kernel/csrc/moe`; byte-for-byte copy into `baseline/` |
| Local edits | None (byte-for-byte copies; `#include <sgl_kernel/...>` resolved via `baseline/include` on the build include path) |

## Python interface → CUDA source mapping

The captured Python interface is `jit_kernel.moe_fused_gate.moe_fused_gate`, i.e. the
**new SGLang JIT-kernel system** under `python/sglang/jit_kernel/`, NOT the legacy
`sgl_kernel.moe.moe_fused_gate` C++ op (that legacy op has a different signature —
`num_expert_group, topk_group, ...` — and is the wrong source; it was not copied).

| Layer | File |
|-------|------|
| Python wrapper | `python/sglang/jit_kernel/moe_fused_gate.py` → copied to `baseline/reference/moe_fused_gate.py` |
| CUDA kernel (header-only, JIT-compiled) | `python/sglang/jit_kernel/csrc/moe/moe_fused_gate.cuh` → copied to `baseline/csrc/moe/moe_fused_gate.cuh` |
| Shared headers | `python/sglang/jit_kernel/include/sgl_kernel/*` → copied to `baseline/include/sgl_kernel/*` (top-level headers + `impl/norm.cuh`; the unrelated `deepseek_v4/` and `distributed/` subtrees were pruned — verified no copied header references them) |

The upstream wrapper loads the kernel via
`load_jit("moe_fused_gate", cuda_files=["moe/moe_fused_gate.cuh"], cuda_wrappers=[("moe_fused_gate", "MoEFusedGateKernel::run")])`
and calls
`module.moe_fused_gate(input, bias, output, indices, topk, scoring_func_int, num_fused_shared_experts, renormalize, routed_scaling_factor, apply_routed_scaling_factor_on_output)`.

### Exact ABI (C++ entry point)
`MoEFusedGateKernel::run`, to be exported as the typed function `moe_fused_gate`:

```
run(TensorView input,    // [num_rows, num_experts] float32  (arg[0])
    TensorView bias,     // [num_experts]          float32  (arg[1])
    TensorView output,   // [num_rows, topk]       float32   <- caller-preallocated
    TensorView indices,  // [num_rows, topk]       int32     <- caller-preallocated
    uint32_t topk,
    uint32_t scoring_func,                 // 0 = sigmoid, 1 = sqrtsoftplus
    uint32_t num_fused_shared_experts,
    bool     renormalize,
    float    routed_scaling_factor,
    bool     apply_routed_scaling_factor_on_output)
```

Inputs first, then the two caller-preallocated outputs (destination-passing), then scalar
params. The Python wrapper allocates `output` (`[num_rows, topk]` float32) and `indices`
(`[num_rows, topk]` int32). Output-allocation policy: **caller-allocated, destination-passing**.

## Captured configuration (this task)

All 296 captured variants use the identical scalar configuration:
`topk=5`, `scoring_func='sigmoid'` (→ `scoring_func=0`), `num_fused_shared_experts=1`,
`renormalize=True`, `routed_scaling_factor=2.0`, `apply_routed_scaling_factor_on_output=True`.
Inputs: `input=[M,128]` float32 contiguous, `bias=[128]` float32 contiguous. So
`num_experts=128`, `topk_routed = topk - num_fused_shared_experts = 4`.

## Recovered kernel semantics (authoritative — from `moe_fused_gate.cuh`)

Per token row:
1. For each expert `e`: `score = compute_score(input[row,e])`; for sigmoid,
   `score = 1/(1+expf(-input))`. `biased = score + bias[e]`.
   - **Selection** uses `biased`; the emitted **weight** uses the un-biased `score`
     (standard DeepSeek-style bias-correction: bias steers selection only).
2. Iterative arg-max selects `topk_routed = topk - num_fused_shared_experts` routed
   experts from `biased`: pick the max, record it, set that slot to `-FLT_MAX`, repeat.
   Selected experts are emitted in **descending biased-score order** (selection order).
   - **Tie-break = smaller expert index wins (large-token path, source-guaranteed).**
     The large-token kernel (`num_rows > 512`, used for prefill M=1074..7432) makes this
     explicit in the warp reduction (`other_val == max_val && other_expert < max_expert`).
   - **Small-token path (`num_rows <= 512`, used for decode M=1..79): tie-break is NOT
     source-guaranteed** — see the "Independent semantics cross-check" section below. For
     random float32 sigmoid+bias scores exact ties are practically never hit, so this only
     matters for the constructed adversarial-tie rows, which are validated empirically on
     GPU (baseline-vs-oracle) rather than assumed.
3. `routed_sum = sum of the topk_routed selected un-biased scores`.
4. Output slot `j` for `j in [0, topk)`:
   - `is_shared = j >= topk_routed` (the last `num_fused_shared_experts` slots).
   - `weight = is_shared ? (routed_sum / routed_scaling_factor) : score_of_selected[j]`.
   - `expert_id = is_shared ? (num_experts + (j - topk_routed)) : selected_expert[j]`.
   - `scale = apply_routed_scaling_factor_on_output ? routed_scaling_factor : 1.0`.
   - `norm  = (renormalize && routed_sum > 0) ? routed_sum : 1.0`.
   - `output[row, j] = weight / norm * scale`;  `indices[row, j] = expert_id`.

### Closed form for this task's config (topk=5, shared=1, renorm=True, rsf=2.0, apply_on_output=True)
- Routed slots `j ∈ {0,1,2,3}` (descending biased order): `index = selected_expert[j]`,
  `weight = score[selected_expert[j]] / routed_sum * 2.0`.
- Shared slot `j = 4`: `index = num_experts = 128`,
  `weight = (routed_sum / 2.0) / routed_sum * 2.0 = 1.0` (when `routed_sum > 0`).

## Supported domain / grouping resolution (AC-1.1)

The `jit_kernel.moe_fused_gate.moe_fused_gate` interface **has no `num_expert_group` /
`topk_group` parameters** — neither the Python wrapper signature nor the C++
`MoEFusedGateKernel::run` accepts them. This is a **flat (ungrouped) top-k over all
`num_experts` experts**; there is no hierarchical 2-layer group limiting in this kernel.
(The legacy `sgl_kernel.moe_fused_gate` C++ op DID take `num_expert_group`/`topk_group`;
that is a different op and is not the captured interface.)

Therefore the grouping question is resolved by the source itself: **grouping is not
applicable** (equivalently `num_expert_group = 1`, `topk_group = 1`). Evidence: the
recovered `moe_fused_gate.cuh` selects directly over all experts with no group mask, and
`baseline/reference/moe_fused_gate.py` has no grouping arguments.

Kernel hard limits (from `MoEFusedGateKernel::run` `RuntimeCheck`s): `num_experts <= 512`,
`scoring_func <= 1`, `topk > num_fused_shared_experts`. The kernel also uses
`kMaxTopK = 16` and a small/large dispatch threshold of `num_rows = 512`.

The candidate's specialized fast-path supported domain is frozen to the captured config:
`num_experts == 128`, `topk == 5`, `scoring_func == sigmoid (0)`,
`num_fused_shared_experts == 1`, `renormalize == true`, `routed_scaling_factor == 2.0`,
`apply_routed_scaling_factor_on_output == true`. Anything outside this domain falls back to
the verbatim-recovered baseline kernel.

## Copied files
```
baseline/csrc/moe/moe_fused_gate.cuh          (kernel; MoEFusedGateKernel::run)
baseline/include/sgl_kernel/atomic.cuh
baseline/include/sgl_kernel/cta.cuh
baseline/include/sgl_kernel/ffi.h
baseline/include/sgl_kernel/impl/norm.cuh
baseline/include/sgl_kernel/math.cuh
baseline/include/sgl_kernel/runtime.cuh
baseline/include/sgl_kernel/scalar_type.hpp
baseline/include/sgl_kernel/source_location.h
baseline/include/sgl_kernel/tensor.h
baseline/include/sgl_kernel/tile.cuh
baseline/include/sgl_kernel/type.cuh
baseline/include/sgl_kernel/utils.cuh
baseline/include/sgl_kernel/utils.h
baseline/include/sgl_kernel/vec.cuh
baseline/include/sgl_kernel/warp.cuh
baseline/reference/moe_fused_gate.py          (upstream Python wrapper, reference only)
```

The build exposes the baseline through the workspace-owned local ABI (`bench/_jit_build.py`,
TVM-FFI `load_inline`) pointing at `baseline/csrc/moe/moe_fused_gate.cuh` with
`baseline/include` on the include path — never importing a live SGLang install.

## Independent semantics cross-check (Codex, gpt-5.5:high)

The recovered semantics above were independently cross-checked against the source
(`VERDICT: CONFIRMED-WITH-CORRECTIONS`). Confirmed: no grouping params (flat top-k);
bias-correction (select on `biased`, weight on un-biased `score`); shared index = 128;
dispatch threshold `num_rows <= 512`; `routed_sum == 0` → `norm = 1.0` (shared weight 0).
Two corrections affect the correctness oracle and must be honored:

1. **Small-token kernel uninitialized-read hazard (decode path, `num_experts == 128`).**
   `warps_per_token = div_ceil(128, 32) = 4`, but `dispatch_small_token_kernel` instantiates
   the `<8>` template, so `moe_fused_gate_kernel_small_token` declares `warp_maxs[8]` /
   `warp_experts[8]` while only warps 0..3 write entries 0..3. Warp 0's final cross-warp
   reduction then reads the **unwritten** `warp_maxs[4..7]` / `warp_experts[4..7]`
   (`moe_fused_gate.cuh:118-126`). Consequence: the small/decode path's selection — and
   therefore its smallest-index tie-break and descending order — is **not source-guaranteed**
   deterministic for this shape. For random float32 inputs exact ties are measure-zero and
   the hazard is expected to be benign in practice, but this is a property to **verify
   empirically on GPU** (baseline-vs-independent-oracle over many seeds), not to assume.
   Tracked as a queued correctness risk; resolved at the correctness-validation step.
   The candidate kernel will implement the well-defined intended semantics (clean top-k,
   smaller-index tie-break, no uninitialized reads).

2. **Shared-slot weight is not bit-exactly 1.0 for subnormal `routed_sum`.** The closed form
   `(routed_sum / 2.0) / routed_sum * 2.0` equals `1.0` for normal positive finite
   `routed_sum`, but a positive *subnormal* `routed_sum` can underflow in `routed_sum / 2.0`.
   The oracle must compute the actual float32 operation order
   `(routed_sum / f32(2.0)) / norm * f32(2.0)` (with `norm = routed_sum if routed_sum > 0
   else 1.0`) rather than hardcoding `1.0`. Routed-slot weights likewise follow
   `score / norm * scale`.

Other oracle edges noted: `+inf`→sigmoid 1, `-inf`→0, `NaN`→NaN propagates through
`routed_sum`; small kernel reads `shared_original_scores[expert_id]` before the
`expert_id >= 0` guard (OOB risk on degenerate selection); `M=0` reaches the launch path
with a zero-block grid (no-op); only `topk > num_fused_shared_experts` is range-checked
(our `topk = 5 <= kMaxTopK = 16`, fine).
