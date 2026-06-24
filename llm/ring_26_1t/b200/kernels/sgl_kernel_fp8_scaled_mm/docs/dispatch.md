# Dispatch Table — `sgl_kernel.fp8_scaled_mm` candidate

Runtime dispatch inspects shape/dtype/stride only (no host sync, no hot-path
allocation). A route-diagnostic (`fp8_scaled_mm_candidate_route`) returns 1 for
the fast path and 0 for fallback and is asserted in `bench/correctness.py` so a
silent fallback can never masquerade as a candidate run.

| Bucket (predicate) | Kernel | Rationale |
|---|---|---|
| `M==1` AND column-major B AND bf16 out AND fp8e4m3 A/B AND `K%16==0` AND **NOT (`K>=4096 && N>=3072`)** | **`fp8_gemv_m1_kernel`** (native CUDA FP8 GEMV, one warp per output column) | Decode GEMV streams Bphys[n,:] coalesced; beats the baseline's 64-row MMA tile. Wins all measured covered shapes (1.14–2.97×). |
| `M==1` AND `K>=4096 && N>=3072` | baseline (CUTLASS sm100) | Largest-work M=1 shapes: the scalar-fp8-decode GEMV is latency/instruction-bound and the baseline's tensor-core tiling wins (measured 0.73–0.86× for the GEMV), so fall back. |
| `2 <= M <= 64` | baseline (CUTLASS sm100) | **Evidence-backed no-go**: an SM100 swap-AB CUTLASS GEMM (`solution/fp8_swapab_smallm.cu`, `pr-vllm-27284`) was implemented + correct but loses (geomean 0.85×; NCU pipeline-fill/occupancy bound). Route gated off (`kSmallMSwapAbEnabled=false`); falls back at parity. See results.md. |
| `M > 64` (prefill) | baseline (CUTLASS sm100) | Compute-bound; beating vendor CUTLASS is unlikely. Fallback. |
| Any uncovered dtype / layout / param (non-column-major B, non-bf16 out, bias, K%16!=0, bad scales) | baseline (CUTLASS sm100) | Correctness-preserving fallback; matches the upstream input contract. |

The `K>=4096 && N>=3072` exclusion is an evidence-backed threshold: both measured
GEMV losers (k8192_n3072 0.86×, k8192_n4608 0.73×) satisfy it; all measured
winners (k8192_n512, k8192_n1024, k8192_n2112, k2304_n8192, k1024_n8192,
k256_n8192, k1536_n1536) do not. Re-tune if the kernel's decode-efficiency
improves (see results.md follow-ups).

## Predicate safety (hardened in Round 1)

`covers_m1_gemv` validates, before returning route 1: exact fp8_e4m3fn DLPack
dtype code (`kDLFloat8_e4m3fn`, not just any 8-bit type — excludes uint8/int8/
e5m2), bf16 out + fp32 scales, A row-major / B column-major / out row-major
layout, `K % 16 == 0`, scale_a `[M,1]` and scale_b `[N,1]` contiguous, out `[M,N]`,
and all tensors on one CUDA device. Additionally, both the baseline export and the
candidate fallback re-validate dtypes at the TensorView boundary
(`require_fp8_contract`) so a contract-violating input is rejected, never
reinterpreted. `bench/correctness.py:negative_route_cases()` proves route==0 (and
baseline rejection) for e5m2/uint8 A, malformed scale ranks `[M,2]`/`[N,2]`,
fp16-out, and mixed-device inputs — each would have misrouted before the Round 1
hardening. Fallback median overhead is +0.009% over 279 uncovered shapes
(host-side predicate only; identical device code).
