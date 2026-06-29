# Results — LTX2 Q/K RMSNorm + Split-RoPE (B200, bit-exact)

## Conclusion: GO

The candidate is **bit-wise equal** to the PyTorch eager baseline (`torch.equal`, zero tolerance)
across the entire correctness suite, and is **faster on every production shape** with an
equal-weight **geometric-mean speedup of 2.154×** (eager-replacement) and **no per-shape-family
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
  rounding. Q and K go through the same kernel.

### Why not a fully-fused custom RMSNorm (bounded attempt, this round)

A fully-fused custom-RMSNorm + split-RoPE CUDA kernel (one block per row, natural fp32 block
reduction) was implemented and tested for bit-exactness against the eager oracle on the production
shapes. It is **NOT bit-exact**: 793 mismatched bf16 elements total (e.g. 471 on the 32640-token
d4096 row, 49 on the 1536 d4096 row; mismatches scale with H and token count). This confirms the
characterization finding — a custom fp32 reduction's `rstd` differs from `aten._fused_rms_norm` by
~1 ULP, flipping outputs at bf16 rounding midpoints. Matching it to 0 ULP would require replicating
torch's exact internal RMSNorm reduction (warp-shuffle order/vectorization), which is fragile and
version-locked. The staged ATen RMSNorm avoids this entirely and already wins on every shape, so the
staged path is the chosen deliverable (DEC-4).

## Correctness (zero tolerance)

`python bench/correctness.py` on B200: **failures=0, skipped=0** across all five sections — 14
production rows (Section 1), 12-row regression grid (Section 2), adversarial rounding-boundary stage
test (Section 3, sensitivity guard tripped on 3419 elems), 25 reject tests incl. odd `head_dim`
(poison-preservation), CPU/wrong-shape norm weight, and mutate-after-accept (Section 4), and
support-helper (Section 5). Comparison is int16-bitcast `torch.equal`. `--rejects-only` passes on CPU.

## Performance (eager-replacement, GPU-event median; `baseline_median / candidate_median`)

| Workload | baseline µs | candidate µs | speedup |
|----------|-------------|--------------|---------|
| ltx23_stage1_video_self_q1536_k1536_d4096 (hd128) | 227.0 | 149.0 | 1.52× |
| ltx23_stage1_audio_self_q126_k126_d2048 (hd64) | 206.9 | 63.0 | 3.28× |
| ltx23_stage1_audio_to_video_q1536_k126_d2048 | 206.7 | 59.4 | 3.48× |
| ltx23_stage1_video_to_audio_q126_k1536_d2048 | 234.6 | 64.5 | 3.63× |
| ltx23_stage2_video_self_q6144_k6144_d4096 (hd128) | 416.9 | 275.7 | 1.51× |
| ltx23_stage2_audio_self_q126_k126_d2048 | 207.7 | 63.6 | 3.27× |
| ltx23_stage2_audio_to_video_q6144_k126_d2048 | 207.6 | 83.5 | 2.49× |
| ltx23_stage2_video_to_audio_q126_k6144_d2048 | 208.1 | 83.8 | 2.49× |
| ltx23_hq_pr29399_stage1_video_self_q8160_k8160_d4096 (hd128) | 539.8 | 352.6 | 1.53× |
| ltx23_hq_pr29399_stage1_audio_to_video_q8160_k126_d2048 | 227.1 | 104.3 | 2.18× |
| ltx23_hq_pr29399_stage1_video_to_audio_q126_k8160_d2048 | 213.6 | 103.9 | 2.06× |
| ltx23_hq_pr29399_stage2_video_self_q32640_k32640_d4096 (hd128) | 1909.8 | 1350.8 | 1.41× |
| ltx23_hq_pr29399_stage2_audio_to_video_q32640_k126_d2048 | 543.7 | 357.0 | 1.52× |
| ltx23_hq_pr29399_stage2_video_to_audio_q126_k32640_d2048 | 589.2 | 361.8 | 1.63× |

- **Headline (equal-weight geomean over the 14 production rows): 2.154×.** Arithmetic mean 2.286×,
  min 1.414×, max 3.635×. **All 14 rows pass correctness and win; none regress.**
- Per-workload median/mean/std/min/p10/p90 are in `bench/results_summary.json` (committed) and the raw
  `bench/results.jsonl` (local).

## Bottleneck analysis (roofline-style, from the measured scaling)

- **Tiny audio / cross rows (S≈126):** the eager baseline is flat at ~207–235 µs regardless of tensor
  size — direct evidence it is **launch / host-overhead bound** (dominated by per-ATen-op CPU launch
  cost, not bytes). The candidate collapses the eager RoPE's several ATen launches into one kernel
  (plus the ATen RMSNorm), landing at ~59–104 µs → 2.1–3.6× there.
- **Large video-self rows (hd128):** the candidate scales ~linearly with token count
  (149→276→353→1351 µs for S = 1536→6144→8160→32640) → **bandwidth / throughput bound**. The win
  (~1.41–1.53×) comes from replacing the eager RoPE ATen ops with one streaming kernel. The remaining
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
  `REMOTE_GPU_ID=5`. Idle before (0% util, 0 MiB) and after (0% util, 4 MiB); GPUs were chosen idle
  and used consistently for correctness and benchmark.
- Python 3.12.3, torch 2.11.0+cu130, CUDA 13.0 (nvcc 13.0.88), host c++ 13.3.0, tvm_ffi 0.1.9.
- Candidate source sha256 `5b5c85d9…`; baseline commit `aaa31eb0…` (copied from SGLang `main`).
- Benchmark: `bench/benchmark.py` (standard template), isolated subprocesses, warmup_runs=10,
  num_trials=7, inner_iterations 1..4096, target_sample_us=1000, A/B interleaved, CUDA-event GPU time.
  Bit-exactness gate runs before timing. No SGLang import/patch at runtime; no `--use_fast_math`; not
  the closed PR #29399 Triton kernel.

## Notes on measurement

The eager baseline's tiny/cross rows are launch/host-overhead bound (~207–235 µs) and therefore
sensitive to host-CPU contention on the shared box; the candidate (one GPU-bound kernel per side) is
stable. Both sides are measured under identical settings in the same interleaved run, so each A/B
ratio is fair. (A prior run measured geomean 2.128×; this run 2.154× — the small cross-run spread is
the contention-sensitive eager baseline, reported honestly.)
