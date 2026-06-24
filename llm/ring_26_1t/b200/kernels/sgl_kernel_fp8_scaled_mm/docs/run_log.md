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

## Round 2 — ABI device guard, expected-route, bias edge, small-M swap-AB no-go

GPU 3 idle pre+post; correctness on GPU 3+4 (mixed-device negative). Candidate
sha256 updated (swap-AB added); see git for the exact hash at HEAD.

Commands (container):
```
CUDA_VISIBLE_DEVICES=3,4 python bench/correctness.py     # 299/299 (286 prod + 4 edge + 9 negatives incl. bias_edge, CPU/mixed-device)
# small-M swap-AB benchmark (route 2 temporarily enabled):
CUDA_VISIBLE_DEVICES=3 python bench/benchmark.py --device cuda:0 --only m{23,32,57}_k{1024,256,8192}_n{8192,512,1024,3072}
# NCU on the swap-AB kernel:
CUDA_VISIBLE_DEVICES=3 ncu --metrics dram__bytes_read...,sm__throughput...,sm__pipe_tensor_cycles_active...,sm__warps_active... --launch-skip 25 -c 1 python bench/prof_one.py candidate 32 1024 8192
```

### Results
- Correctness 299/299 (adds CUDA/same-device guard tests that CALL the candidate,
  expected-route assertions for every row, and a real bias edge vs the fp32 oracle).
- **Small-M swap-AB benchmark** (m23/m32/m57 × 5 hot K,N): every shape 0.83–0.88×,
  geomean **0.85×** (candidate ~20–21µs vs baseline ~17–18µs). Not one ≥1.10.
- **NCU (swap-AB kernel, m32·k1024·n8192)**: tensor-core 1.9%, SM throughput 6.5%,
  occupancy 10.7%, DRAM 10.7% → pipeline-fill / occupancy bound (tiny swapped-N=M
  under-fills the 2-SM warp-specialized mainloop).
- **Verdict**: small-M = evidence-backed no-go; route gated off
  (`kSmallMSwapAbEnabled=false`), falls back to baseline (no regression). M=1
  promotion unchanged. Full small-M table + bound in `docs/results.md`.

## Round 3 — real bias candidate-fallback route + final provenance

GPU 3 idle; correctness on GPU 3+4. Final source hashes at HEAD:
- `solution/fp8_scaled_mm_candidate.cu`: `507dcce3532d30f35fabc08d4ecdb220f3ffd9f03fcc9939d5be1d9d3a3a99ee`
- `baseline/fp8_scaled_mm_baseline.cu`: `69979be8ec322ab2580ef1d356a61b75f749fb13af9f5ca27c5d39546c96a5bf`
- `solution/fp8_swapab_smallm.cu`: `d19e004ed0e4c579049b48b4b9d2575ab8c7d0e6324a6ef08b2f0172fbbb7c19`
- `bench/correctness.py`: `8338da181d4c6ac6d6c2e64c5c614b5cd8a8d6e6418693735cdd7b7dd10ca76b`
- `bench/benchmark.py` == template `2e1712e567b50ba19340f62314f13e122fc3b547775bd06ed6c4d284d144ee13`

Final correctness command + result:
```
CUDA_VISIBLE_DEVICES=3,4 python bench/correctness.py
=> 299 passed, 0 failed (286 production + 4 edge + 9 negative/edge)
```
The bias edge now runs through the candidate fallback route: `route_bias=0`,
`baseline_bias` and `candidate_bias` both match the fp32 oracle (`out=(A@B)·
scale_a·scale_b+bias`), `candidate==baseline`. Benchmark numbers (M=1 promotion,
small-M no-go, fallback overhead) are unchanged from Rounds 1–2 (the candidate's
fast-path code is unchanged; only the test-only bias entry was added).

## Round 4 — code-review P2 fixes (M=1 fast path)

Fixed two [P2] code-review findings in the candidate fast path (no verdict/
benchmark change; the GEMV math is unchanged):
- Predicates now require `b.stride(1)==K` (exactly-packed Bphys) so a padded/sliced
  column-major B falls back to the baseline instead of the GEMV reading wrong
  columns. New `neg_padded_B` test asserts route 0.
- `fp8_scaled_mm_candidate` sets a `c10::cuda::CUDAGuard` to the tensors' device and
  uses `getCurrentCUDAStream(device_id)` before the GEMV launch (right-GPU launch).

Verified on ion-b200 GPU 3+4 (idle): correctness **300/300** (286 production + 4
edge + 10 negatives). Updated source hashes: candidate
`4a1d9a9d3700415f0a4eb18e303a994159ede9e0cfac4847ffb822c0044b636b`,
`bench/correctness.py` `e9549c5af409110c62b8b4bb7addcdb932ae32e1b7c5207ee53028cfe6855f36`
(baseline/swap-AB unchanged).

## Round 5 — production-only workloads.json (benchmark harness fix)

Code-review [P2]: the default `python bench/benchmark.py` ran every row in
workloads.json, including the row-major `edge_contigB`, which the baseline rejects
(raise) -> the benchmark exited ERROR before producing full-grid results. Fix:
`workloads.json` is now the 286 production rows only; the 4 edge rows moved to the
correctness-only `workloads_edges.json` (read by `correctness.py`). The full-grid
benchmark can now run on the default input directly:
```
CUDA_VISIBLE_DEVICES=3 python bench/benchmark.py --device cuda:0   # default workloads.json = 286 production
```
Verified on ion-b200 GPU 3 (idle): default `bench/benchmark.py` runs clean
(workloads.json has no non-production / no row-major B); correctness 300/300
(edge_contigB still rejected, now via workloads_edges.json). Updated hashes:
`bench/correctness.py` `19d0b19a2e1273efa993bc190418daf716ae33303b58b878b3fc65614db6310f`,
`bench/gen_workloads.py` `02b86df01ef74e9c91e28a286f8561c529b50fb9c9d3904acb612ea6c139000e`
(candidate/baseline/swap-AB unchanged).

## Round 6 — destination-out validation + N-row alignment (edge-contract fixes)

Two code-review [P2] fixes (no verdict/benchmark change; no production impact):
- `fp8_scaled_mm_baseline_impl` validates the caller-provided `out` (shape
  [M,N], row-major, 16-byte-aligned row) before the CUTLASS dispatch (packed D
  strides) — a short/strided out is now rejected, not written OOB.
- `covers_m1_gemv` + `covers_smallm_swapab` require `(N*2)%16==0` (N%8==0 for bf16),
  so an unaligned-N M=1 shape (which the upstream rejects) falls back to baseline.

Verified on ion-b200 GPU 3+4 (idle): correctness **302/302** (290 workload rows +
12 negatives, incl. `neg_unaligned_N` + `neg_bad_out`, both route 0 +
baseline-rejected). Updated hashes: candidate
`c7d4fc3f05caff28ba63e1c93edf3f21f217aae891989a0de97174faf324d1df`, baseline
`18f28aff1ec165333d9a4da03c4629d1cd0bf139d49caf728aff29e7588b906b`,
`bench/correctness.py` `0cc75d647875c10dfe67157e73e804266f2d9afcb75b661974229f23a725997e`
(swap-AB unchanged).
