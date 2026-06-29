# Results — LTX2 Q/K RMSNorm + Split-RoPE (B200, bit-exact)

## Conclusion: GO

The candidate is **bit-wise equal** to the PyTorch eager baseline (`torch.equal`, zero
tolerance) across the entire correctness suite, and is **faster on every production shape**
with an equal-weight **geometric-mean speedup of 2.13×** (eager-replacement) and **no
per-shape-family regression**.

## Approach (staged, bit-exact)

- RMSNorm over the full hidden `H`: **ATen `torch.nn.RMSNorm`** (the eager module itself),
  bit-exact by construction.
- Split-RoPE: one custom CUDA kernel (`solution/kernel.cu`, sm_100) per side, reproducing the
  eager rounding sequence exactly — `bf16(x*cos)` rounded first, then the sine term combined in
  fp32 with a single final bf16 cast (variant A1), cos/sin indexed via real (non-contiguous)
  strides, explicit `__fmul_rn`/`__fadd_rn`/`__float2bfloat16_rn` so `-O3` cannot contract the
  visible rounding. Q and K go through the same kernel.
- Why staged: the B200 probe (`docs/numerics_characterization.md`) showed a naive custom fp32
  RMS reduction's `rstd` is ≤1 ULP off `aten._fused_rms_norm`, flipping ~25/2.4M output bf16
  values. ATen RMSNorm avoids that while the fused RoPE kernel removes the eager RoPE's multiple
  ATen launches. A fully-fused custom RMSNorm (to also drop the intermediate-tensor HBM traffic)
  is a future optimization gated on a 0-ULP rstd match.

## Correctness (zero tolerance)

`python bench/correctness.py` on B200: **failures=0, skipped=0** across all five sections —
14 production rows (Section 1), 12-row regression grid (Section 2), adversarial rounding-boundary
stage test (Section 3, sensitivity guard tripped on 3419 elems), 22 reject tests incl.
mutate-after-accept (Section 4), and support-helper (Section 5). Comparison is int16-bitcast
`torch.equal`. `--rejects-only` passes on CPU.

## Performance (eager-replacement, GPU-event median; `baseline_median / candidate_median`)

| Workload | baseline µs | candidate µs | speedup |
|----------|-------------|--------------|---------|
| ltx23_stage1_video_self_q1536_k1536_d4096 (hd128) | 220.73 | 148.38 | 1.49× |
| ltx23_stage1_audio_self_q126_k126_d2048 (hd64) | 187.74 | 51.60 | 3.64× |
| ltx23_stage1_audio_to_video_q1536_k126_d2048 | 196.24 | 53.46 | 3.67× |
| ltx23_stage1_video_to_audio_q126_k1536_d2048 | 200.78 | 54.69 | 3.67× |
| ltx23_stage2_video_self_q6144_k6144_d4096 (hd128) | 412.73 | 275.39 | 1.50× |
| ltx23_stage2_audio_self_q126_k126_d2048 | 194.40 | 54.18 | 3.59× |
| ltx23_stage2_audio_to_video_q6144_k126_d2048 | 198.43 | 83.39 | 2.38× |
| ltx23_stage2_video_to_audio_q126_k6144_d2048 | 189.51 | 83.18 | 2.28× |
| ltx23_hq_pr29399_stage1_video_self_q8160_k8160_d4096 (hd128) | 526.58 | 349.49 | 1.51× |
| ltx23_hq_pr29399_stage1_audio_to_video_q8160_k126_d2048 | 189.39 | 103.63 | 1.83× |
| ltx23_hq_pr29399_stage1_video_to_audio_q126_k8160_d2048 | 194.59 | 103.49 | 1.88× |
| ltx23_hq_pr29399_stage2_video_self_q32640_k32640_d4096 (hd128) | 1903.81 | 1341.09 | 1.42× |
| ltx23_hq_pr29399_stage2_audio_to_video_q32640_k126_d2048 | 554.75 | 355.90 | 1.56× |
| ltx23_hq_pr29399_stage2_video_to_audio_q126_k32640_d2048 | 574.62 | 358.22 | 1.60× |

- **Headline (equal-weight geomean over the 14 production rows): 2.128×.** Arithmetic mean 2.286×,
  min 1.420×, max 3.671×. **All 14 rows pass correctness and win; none regress.**
- Per-workload median/mean/std/min/p10/p90 and raw samples are in `bench/results.jsonl`.

## Bottleneck analysis (roofline-style, from the measured scaling)

- **Tiny audio / cross rows (S≈126):** the eager baseline is flat at ~187–200 µs regardless of
  tensor size — direct evidence it is **launch / host-overhead bound** (dominated by per-ATen-op
  CPU launch cost, not bytes). The candidate collapses the eager RoPE's several ATen launches into
  one kernel (plus the ATen RMSNorm), landing at ~52–104 µs → 2.3–3.7× there.
- **Large video-self rows (hd128):** the candidate scales ~linearly with token count
  (148→275→349→1341 µs for S = 1536→6144→8160→32640) → **bandwidth / throughput bound**. The win
  (~1.42–1.51×) comes from replacing the eager RoPE ATen ops with one streaming kernel. The
  remaining headroom is the staged intermediate normed-tensor HBM traffic; a fully-fused custom
  RMSNorm would remove it (future optimization, gated on the rstd 0-ULP match — see DEC-4).
- No specialization/dispatcher is used: a single general kernel wins on every shape family, so per
  the source prompt a dispatcher is unnecessary (it is required only if one generic kernel cannot
  win across CI {1536,6144} and HQ {8160,32640}, which it does). NCU SASS-level profiling is
  available (`external/ncu-report-skill`) if a future fused-RMSNorm round needs achieved-bandwidth
  evidence; it is not needed to justify this clear, no-regression win.

## Provenance / environment (GPU idle-state verified)

- Host `ion-b200` (`innomatrix-us-adc-smb200-0003`), container `sglang_bbuf_pr29315`,
  workspace `/tmp/ltx2_qknorm_task`.
- GPU **id 5**, NVIDIA B200 (sm_100), pinned via `CUDA_VISIBLE_DEVICES=5`. Idle before
  (0% util, ≤4 MiB residual) and after (0% util, 4 MiB); GPUs 1–4 were busy, so id 5/6 were the
  idle choices and id 5 was used consistently for correctness and benchmark.
- Python 3.12.3, torch 2.11.0+cu130, CUDA 13.0, tvm_ffi 0.1.9, nvcc 13.0.
- Benchmark: `bench/benchmark.py` (standard standalone template), isolated subprocesses,
  warmup_runs=10, num_trials=7, inner_iterations 1..4096, target_sample_us=1000, A/B interleaved,
  CUDA-event GPU time. Bit-exactness gate runs before timing.
- Baseline = PyTorch eager fallback (`baseline/`, copied from SGLang `main` `aaa31eb0…`); no SGLang
  import/patch at runtime. Compile flags symmetric, no `--use_fast_math`; not the closed PR #29399
  Triton kernel.

## Notes on measurement

The eager baseline's tiny/cross rows are launch/host-overhead bound (~187–200 µs) and therefore
sensitive to host-CPU contention on the shared box; the candidate (one GPU-bound kernel per side)
is stable. Both sides are measured under identical settings in the same interleaved run, so each
A/B ratio is fair.
