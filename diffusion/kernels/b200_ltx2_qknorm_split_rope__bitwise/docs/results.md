# Results — LTX2 Q/K RMSNorm + Split-RoPE (B200, bit-exact)

## Conclusion: GO

The candidate is **bit-wise equal** to the PyTorch eager baseline (`torch.equal`, zero tolerance)
across the entire correctness suite, and is **faster on every production shape** with an
equal-weight **geometric-mean speedup of 2.113×** (eager-replacement) and **no per-shape-family
regression**.

Committed evidence artifact: **`bench/results_summary.json`** (compact: full provenance + per-row
median/mean/std/min/p10/p90 + headline; no raw samples). The raw `bench/results.jsonl` is kept
locally on the remote workspace (gitignored) per PR-scope hygiene.

## Approach (staged, bit-exact)

- RMSNorm over the full hidden `H`: **ATen `torch.nn.RMSNorm`** (the eager module itself), bit-exact
  by construction.
- Split-RoPE: one custom CUDA kernel (`solution/kernel.cu`, sm_100) per side, reproducing the eager
  rounding sequence exactly — `bf16(x*cos)` rounded first, then the sine term combined in fp32 with a
  single final bf16 cast (variant A1), cos/sin indexed via real (non-contiguous) strides, explicit
  `__fmul_rn`/`__fadd_rn`/`__fsub_rn`/`__float2bfloat16_rn` so `-O3` cannot contract the visible
  rounding. Q and K go through the same kernel. The entry point sets an `at::cuda::CUDAGuard` on the
  input tensor's device, so the stream/launch target the data's device even on a non-current GPU
  (multi-GPU safe).

### Why not a fully-fused custom RMSNorm (bounded attempt)

A fully-fused custom-RMSNorm + split-RoPE CUDA kernel (one block per row, natural fp32 block
reduction) was implemented and tested for bit-exactness against the eager oracle on the production
shapes. It is **NOT bit-exact**: 793 mismatched bf16 elements total (e.g. 471 on the 32640-token
d4096 row, scaling with H and token count) — a custom fp32 reduction's `rstd` differs from
`aten._fused_rms_norm` by ~1 ULP, flipping outputs at bf16 rounding midpoints. Matching it to 0 ULP
would require replicating torch's exact internal RMSNorm reduction (warp-shuffle order/vectorization),
fragile and version-locked. **The staged ATen RMSNorm is the user-approved final scope (DEC-4).**

## Correctness (zero tolerance)

`python bench/correctness.py` on B200: **failures=0, skipped=0** across all five sections — 14
production rows (Section 1), 12-row regression grid (Section 2), adversarial rounding-boundary stage
test (Section 3, sensitivity guard tripped on 3419 elems), 26 reject tests incl. odd `head_dim`
(poison-preservation), CPU / wrong-shape / missing-affine (`elementwise_affine=False`) norm weight,
and mutate-after-accept (Section 4), and support-helper (Section 5). Comparison is int16-bitcast
`torch.equal`. `--rejects-only` passes on CPU.

## Performance (eager-replacement, GPU-event median; `baseline_median / candidate_median`)

| Workload | baseline µs | candidate µs | speedup |
|----------|-------------|--------------|---------|
| ltx23_stage1_video_self_q1536_k1536_d4096 (hd128) | 220.0 | 148.9 | 1.48× |
| ltx23_stage1_audio_self_q126_k126_d2048 (hd64) | 190.6 | 54.0 | 3.53× |
| ltx23_stage1_audio_to_video_q1536_k126_d2048 | 193.9 | 55.2 | 3.51× |
| ltx23_stage1_video_to_audio_q126_k1536_d2048 | 187.6 | 54.4 | 3.45× |
| ltx23_stage2_video_self_q6144_k6144_d4096 (hd128) | 408.8 | 273.6 | 1.49× |
| ltx23_stage2_audio_self_q126_k126_d2048 | 187.0 | 54.3 | 3.44× |
| ltx23_stage2_audio_to_video_q6144_k126_d2048 | 186.0 | 82.8 | 2.25× |
| ltx23_stage2_video_to_audio_q126_k6144_d2048 | 203.2 | 83.4 | 2.44× |
| ltx23_hq_pr29399_stage1_video_self_q8160_k8160_d4096 (hd128) | 532.4 | 351.7 | 1.51× |
| ltx23_hq_pr29399_stage1_audio_to_video_q8160_k126_d2048 | 189.3 | 103.7 | 1.83× |
| ltx23_hq_pr29399_stage1_video_to_audio_q126_k8160_d2048 | 213.3 | 104.1 | 2.05× |
| ltx23_hq_pr29399_stage2_video_self_q32640_k32640_d4096 (hd128) | 1904.3 | 1339.1 | 1.42× |
| ltx23_hq_pr29399_stage2_audio_to_video_q32640_k126_d2048 | 543.2 | 357.2 | 1.52× |
| ltx23_hq_pr29399_stage2_video_to_audio_q126_k32640_d2048 | 588.5 | 361.7 | 1.63× |

- **Headline (equal-weight geomean over the 14 production rows): 2.113×.** Arithmetic mean 2.253×,
  min 1.422×, max 3.527×. **All 14 rows pass correctness and win; none regress.**
- Per-workload median/mean/std/min/p10/p90 are in `bench/results_summary.json` (committed) and the raw
  `bench/results.jsonl` (local).

## Bottleneck analysis (roofline-style, from the measured scaling)

- **Tiny audio / cross rows (S≈126):** the eager baseline is flat at ~186–203 µs regardless of tensor
  size — direct evidence it is **launch / host-overhead bound** (dominated by per-ATen-op CPU launch
  cost, not bytes). The candidate collapses the eager RoPE's several ATen launches into one kernel
  (plus the ATen RMSNorm), landing at ~54–104 µs → 1.8–3.5× there.
- **Large video-self rows (hd128):** the candidate scales ~linearly with token count
  (149→274→352→1339 µs for S = 1536→6144→8160→32640) → **bandwidth / throughput bound**. The win
  (~1.42–1.51×) comes from replacing the eager RoPE ATen ops with one streaming kernel. The remaining
  headroom is the staged intermediate normed-tensor HBM traffic; a fully-fused custom RMSNorm would
  remove it but cannot be made bit-exact (see the bounded attempt above), so it is out of scope.
- No specialization/dispatcher is used: a single general kernel wins on every shape family, so per the
  source prompt a dispatcher is unnecessary (required only if one generic kernel cannot win across CI
  {1536,6144} and HQ {8160,32640}, which it does). NCU SASS-level profiling is available
  (`external/ncu-report-skill`); it is not needed to justify this clear, no-regression win.

## Provenance / environment (GPU idle-state verified; full record in bench/results_summary.json)

- Host `ion-b200` (`innomatrix-us-adc-smb200-0003`), container `sglang_bbuf_pr29315`, workspace
  `/tmp/ltx2_qknorm_task`.
- GPU **id 5**, NVIDIA B200 (sm_100), uuid `8b907ca0-…`, pinned via `CUDA_VISIBLE_DEVICES=5` /
  `REMOTE_GPU_ID=5`. Idle before (0% util, 4 MiB residual) and after (0% util, 4 MiB); GPUs were
  chosen idle and used consistently for correctness and benchmark.
- Python 3.12.3, torch 2.11.0+cu130, CUDA 13.0 (nvcc 13.0.88), host c++ 13.3.0, tvm_ffi 0.1.9.
- Candidate source sha256 `e482faaa…`; baseline commit `aaa31eb0…` (copied from SGLang `main`).
- Benchmark: `bench/benchmark.py` (standard template), isolated subprocesses, warmup_runs=10,
  num_trials=7, inner_iterations 1..4096, target_sample_us=1000, A/B interleaved, CUDA-event GPU time.
  Bit-exactness gate runs before timing. No SGLang import/patch at runtime; no `--use_fast_math`; not
  the closed PR #29399 Triton kernel.

## Notes on measurement

The eager baseline's tiny/cross rows are launch/host-overhead bound (~186–203 µs) and therefore
sensitive to host-CPU contention on the shared box; the candidate (one GPU-bound kernel per side) is
stable. Both sides are measured under identical settings in the same interleaved run, so each A/B
ratio is fair. (Across runs the geomean has been 2.128× / 2.154× / 2.106× / 2.113× — the small
cross-run spread is the contention-sensitive eager baseline, reported honestly; every row wins in
every run.)
