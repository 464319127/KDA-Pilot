# Baseline Source Provenance — `sgl_kernel.fp8_scaled_mm`

## Upstream lineage

| Field | Value |
|---|---|
| Repository | `https://github.com/sgl-project/sglang` |
| Branch | `main` |
| Resolved commit SHA | `34dd9c28caf4f7dd185e58e462a1344b52568e2e` |
| Resolution time (UTC) | `2026-06-23T23:41:25Z` (local date 2026-06-24) |
| Recovery method | `git ls-remote` to resolve `main` HEAD, then a blobless sparse checkout of `sgl-kernel/` at that exact commit |
| Local edits to copied source | **None** — all files copied verbatim from the pinned commit |

The resolved `main` HEAD at `git ls-remote` time matched the sparse-checkout HEAD exactly (`34dd9c28…`), so the copied source is from that exact commit.

## External build dependency — CUTLASS (vendored via FetchContent)

`sgl-kernel` does not vendor CUTLASS in-tree; it fetches it at build time:

| Field | Value |
|---|---|
| Source | `sgl-kernel/CMakeLists.txt` `FetchContent_Declare(repo-cutlass …)` |
| CUTLASS repo | `https://github.com/NVIDIA/cutlass` |
| CUTLASS commit | `57e3cfb47a2d9e0d46eb6335c3dc411498efa198` |
| Include dirs | `cutlass/include`, `cutlass/tools/util/include` |
| Relevant build defines | `-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_VERSIONS_GENERATED -DCUTLASS_TEST_LEVEL=0` |

CUTLASS is a **build-time dependency only**; it is fetched into a local build-deps location on the remote B200 (not staged into `baseline/` and not committed to the PR — it is large and external). The exact commit above is pinned for reproducibility.

## Copied files (verbatim → `baseline/`)

| Source (at `34dd9c28…`, under `sgl-kernel/`) | Destination |
|---|---|
| `csrc/gemm/fp8_gemm_kernel.cu` | `baseline/fp8_gemm_kernel.cu` |
| `csrc/gemm/math.hpp` | `baseline/math.hpp` |
| `include/utils.h` | `baseline/utils.h` |
| `csrc/cutlass_extensions/` (15 files, full subtree) | `baseline/cutlass_extensions/` |

`fp8_gemm_kernel.cu` sha256: `be6f95bb0ef108ebf08b8b84336b6840726beb7fa5bbb90aa6eb4038ce778767`.

The `cutlass_extensions/` subtree is self-contained: its only non-CUTLASS local includes resolve within `cutlass_extensions/` itself; everything else is CUTLASS/CuTe (from the pinned CUTLASS commit) or torch/cuda_runtime. `fp8_gemm_kernel.cu` includes `"cutlass_extensions/gemm/fp8_gemm_sm90_dispatch.cuh"`, `"math.hpp"`, and `"utils.h"`, all resolvable with `-I baseline`.

> Note: the sm90/sm89/sm120 dispatch headers (and the mixed-input collective files under `cutlass_extensions/gemm/collective/`) are copied because `fp8_gemm_kernel.cu` `#include`s the sm90 dispatch header unconditionally and must compile as-is. On B200 (sm_100) only the **sm100 path** executes at runtime; the other paths are present only so the verbatim file compiles.

## Python interface and entry point

- Python wrapper: `sgl-kernel/python/sgl_kernel/gemm.py::fp8_scaled_mm` — a thin pass-through to `torch.ops.sgl_kernel.fp8_scaled_mm.default(mat_a, mat_b, scales_a, scales_b, out_dtype, bias)`.
- C++ entry: `fp8_scaled_mm(...)` in `fp8_gemm_kernel.cu`. Return-value style (allocates `out = torch::empty({M, N}, out_dtype)` internally). The local ABI converts this to destination-passing (output pre-allocated by the harness, passed last).
- Input contract enforced by the entry: `mat_a` row-major fp8_e4m3 `[M,K]`; `mat_b` **column-major** fp8_e4m3 `[K,N]` (`stride(0)==1`); `scales_a` fp32 contiguous, `numel==M`; `scales_b` fp32 contiguous, `numel==N`; `out_dtype ∈ {bf16, half}`; optional `bias` (`numel==N`, dtype==out_dtype). All match the captured Ring-2.6-1T workload (bf16 out, `bias=None`).

## Recorded dispatch path on sm_100 (B200) — authoritative baseline

`fp8_scaled_mm` selects by `getSMVersion()` (`utils.h`). For B200 (sm_100, `>=100 && <120`), with `CUDA_VERSION >= 12080`, it calls `sm100_fp8_dispatch_shape<cutlass::bfloat16_t>` → `sm100_fp8_dispatch_bias`. This is a **CUTLASS 3.x sm100 row-wise (per-token + per-channel) scaled FP8 GEMM** (`DeviceGemmFp8RowwiseSm100`), NOT DeepGEMM and NOT a torch fallback. Layout: A `RowMajor`, B `ColumnMajor`, D `RowMajor`; epilogue `out = scale_a[m] * (scale_b[n] * acc)` (+bias if present) via `Sm90ColBroadcast`(scale_a) and `Sm90RowBroadcast`(scale_b).

### M-bucketed tile selection (no swap-AB)

`mp2 = max(16, next_pow_2(M))`, then:

| M range (`mp2`) | Config | CTA tile (M,N,K) | Cluster |
|---|---|---|---|
| `mp2 ≤ 16` (M∈[1,16]) | `Gemm16` | `64 × 64 × 128` | `1 × 4 × 1` |
| `mp2 ≤ 64` (M∈(16,64]) | `Gemm64` | `64 × 64 × 128` | `1 × 1 × 1` |
| `mp2 ≤ 256` (M∈(64,256]) | `Gemm256` | `128 × 128 × 128` | `2 × 1 × 1` |
| else (M>256) | `GemmDefault` | `256 × 128 × 64` | `2 × 2 × 1` |

**Key finding for the optimization.** Every small-M config uses a **CTA tile-M of 64** with a standard A-major MMA. For the dominant decode shape M=1 (and the hot M∈{3..57} small-batch shapes), the tensor-core M dimension is filled with 1..57 useful rows out of 64 → the MMA is heavily under-utilized in the decode regime. The sm100 path has **no swap-AB** (which would map the large N onto the MMA's M-axis so tiny real-M is on the tolerant N-axis) — swap-AB exists upstream only in the **sm90** dispatch header (`cutlass_extensions/gemm/fp8_gemm_sm90_dispatch.cuh`) and in `sgl-kernel/benchmark/bench_fp8_gemm_swap_ab.py`, not on sm100. This is the concrete decode-regime gap the candidate targets (cf. KernelWiki `pr-vllm-27284`, which added SM100 swap-AB for M≤64 to vLLM's CUTLASS FP8 GEMM).

This recorded dispatch path resolves plan decision DEC-6 (authoritative baseline path): the stock default sm_100 dispatch is the CUTLASS row-wise scaled FP8 GEMM above; no env flags alter it for this interface/shape set.

## Runtime isolation

The baseline is exposed through a workspace-owned, locally-built module (see `bench/` ABI). No live SGLang server or installed `sgl_kernel` package is imported, patched, or monkey-patched during correctness or benchmark runs — only the copied source above is compiled and called.
