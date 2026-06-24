# Remote Run Log — `sgl_kernel.fp8_scaled_mm` on B200

All GPU work runs on the pinned remote B200, GPU id 3. No benchmark/profile is
taken unless GPU 3 is verified idle (no active compute, no meaningful memory)
before and after.

## Environment

| Field | Value |
|---|---|
| Host | `ion-b200` (`innomatrix-us-adc-smb200-0003`) |
| Container | `sglang_bbuf` (lmsysorg/sglang:dev, privileged + SYS_ADMIN for ncu) |
| GPU | NVIDIA B200, id **3** (`REMOTE_GPU_ID=3`, pinned; pinned via `CUDA_VISIBLE_DEVICES=3`) |
| Compute capability | sm_100 (10.0) |
| CUDA (nvcc) | 13.0, V13.0.88 (`/usr/local/cuda/bin/nvcc`) |
| torch | 2.11.0+cu130 (CUDA runtime 13.0) |
| tvm_ffi | present (`tvm_ffi.cpp.load`) |
| CUTLASS (build dep) | NVIDIA/cutlass @ 57e3cfb4 (shallow checkout, 184M, in `.deps/cutlass`) |
| Remote workspace | `/home/sglang-omni/bbuf/kda/k04_fp8_scaled_mm` (task-owned; synced from the local worktree, excludes .git/.humanize/.deps/.build) |

## Round 0 — baseline foundation verification (build + correctness + harness)

GPU 3 idle pre **and** post (`nvidia-smi -i 3`: util 0%, mem 0 MiB).

Exact commands (inside container, `CUDA_VISIBLE_DEVICES=3`):

```
bash bench/setup_cutlass.sh                      # fetch CUTLASS@57e3cfb4 -> .deps/cutlass
python -c "import build_ext; build_ext.get_ext()"  # build both sides, one module
python bench/correctness.py                      # full grid + edges
python bench/benchmark.py --device cuda:0 --only m1_k1024_n8192 m23_k8192_n512 m57_k1024_n8192
```

Build flags (symmetric, both sides; provenance): `-std=c++17 -O3
--expt-relaxed-constexpr --expt-extended-lambda -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1
-DCUTLASS_VERSIONS_GENERATED -DCUTLASS_TEST_LEVEL=0 --threads=1 -lineinfo
-gencode=arch=compute_100a,code=sm_100a`. No `--use_fast_math` (neither side).

### Results

- **Build**: OK — baseline TU (`cuda_0.o`, 5.7 MB, the verbatim `fp8_gemm_kernel.cu`
  + all CUTLASS sm100 templates) and candidate TU compiled together into one
  TVM-FFI module (`fp8_scaled_mm_ext`). Baseline TU compile ≈ 6 min.
- **Correctness** (`bench/correctness.py`): **291 passed, 0 failed** (286 production
  shapes + 5 edge rows). Every production row: candidate (identity stub) matches
  the fp32-dequant oracle within bf16 tol (atol 0.07 / rtol 0.02) AND is
  bit-identical to the baseline (route==0 fallback). Edge rows: `bias!=None` →
  fallback, contiguous-B → rejected by the input contract, fp16-out / off-M /
  rare-(K,N) → ok.
- **Timing harness** (`bench/benchmark.py`, 3 hot headline shapes, stub candidate):

  | shape | regime | baseline (µs) | candidate (µs) | speedup |
  |---|---|---|---|---|
  | m1_k1024_n8192 | decode_tiny | 12.220 | 12.197 | 1.0019 |
  | m23_k8192_n512 | decode_small | 10.260 | 10.282 | 0.9979 |
  | m57_k1024_n8192 | decode_small | 10.325 | 10.320 | 1.0006 |

  geomean 1.0001 — as expected (stub candidate == baseline). Confirms the
  CUDA-event + inner-loop-amplification + isolated-subprocess timing path works.

### Preliminary roofline insight (drives the candidate)

For the dominant decode shape **M=1, K=1024, N=8192**: the GEMV must read B
(1024×8192 fp8 ≈ 8.39 MB) + A (1 KB) and write out (8192×bf16 ≈ 16 KB), ≈ 8.4 MB
of HBM traffic. Baseline 12.22 µs ⇒ ≈ **690 GB/s ≈ 8.6% of the B200's ~8 TB/s
HBM** — the upstream sm100 path uses a 64-row MMA tile (Gemm16) even for M=1, so
the decode regime is far from bandwidth-bound-efficient. A bandwidth-optimal FP8
skinny-GEMV that streams B once is the primary candidate direction (to be
confirmed with NCU). (Estimate from wall-clock; NCU `dram__bytes_read.sum.*` to
follow.)

## Round 1 — hardened dispatch, candidate, full-grid evidence

GPU 3 idle pre+post (`nvidia-smi -i 3`: util 0%, mem 0–4 MiB) for every measured
run. Candidate source sha256 `1580070dd641f3f049a02824210c18e5d3ce153287abba5802f7d3cb71c7d950`;
baseline ABI wrapper `07d0403b2382530660e31cd6d0f2ee8047dc3748d78aff1422c0b5880ab7947c`;
benchmark.py byte-identical to the template (sha256 `2e1712e567b50ba19340f62314f13e122fc3b547775bd06ed6c4d284d144ee13`).

Exact commands (container, `CUDA_VISIBLE_DEVICES=3`, except correctness uses `3,4`
so the mixed-device negative test runs):
```
python -c "import build_ext; build_ext.get_ext()"                # rebuild (hardened predicate + ABI guard)
CUDA_VISIBLE_DEVICES=3,4 python bench/correctness.py             # full grid + edges + negatives
python bench/benchmark.py --device cuda:0 --workloads bench/workloads_prod.json --out bench/results_full.jsonl  # all 286
python bench/analyze_results.py bench/results_full.jsonl         # geomeans + significance + overhead
python bench/benchmark.py --device cuda:0 --num-trials 25 --only <6 uncovered shapes>  # fallback overhead
```

### Results
- Correctness: 296 passed / 0 failed (286 production + 4 edge + 6 negative-route:
  e5m2/uint8 A, scale [M,2]/[N,2], fp16-out, mixed-device → all route 0 and
  baseline-rejected where applicable).
- Full-grid (286 production): equal-weight geomean 1.0228, call-weighted 1.1667,
  time-weighted 1.1346. Covered M=1 geomean 2.78× (7/7 significant, candidate p90
  < baseline p10; 0 regressions). Per-regime: decode_tiny 1.117, decode_small
  0.997, medium 0.990, prefill 1.000.
- Fallback overhead: full-grid median +0.009% (279 uncovered shapes); dedicated
  25-trial run on 6 uncovered shapes geomean +0.88% (worst +1.67%).
- Per-shape significance table + analysis in `docs/results.md`.
