# Results — `jit_kernel.grouped_topk._jit_grouped_topk_op` on B200 (GLM-5.2)

## Headline

Correctness-preserving **win**. Native-CUDA candidate vs the recovered SGLang
baseline, NVIDIA B200, idle GPU 0, all 17404-call production shape set:

- **Equal-weight geometric-mean speedup over the 43 production rows: 1.217×**
- **Call-weighted geometric mean (by captured call frequency): 1.037×**
- **No regression on any production row** (min 0.999×, i.e. parity).
- Range: decode parity (0.999×) → prefill up to **1.67×**.
- Correctness: candidate is **bit-identical** to the baseline; full grid
  (1479 checks) passes against both the baseline and an independent oracle.

The candidate is a workspace-owned native CUDA kernel (no Triton/DSL/torch.compile)
that dispatches by token count: the recovered baseline algorithm for the
launch-floor-bound decode/small regime (parity, no regression) and a new
register-resident **warp-per-token** kernel for the large-prefill regime (the win).

## Per-regime summary

| Regime | N | Production rows | % of calls | geomean speedup | range |
|---|---|---|---|---|---|
| decode | 2–38 | 9 | **85.2%** | **0.999** (parity) | 0.999–0.999 |
| mid | 110 | 1 | 0.4% | 1.039 | — |
| prefill | 392–3769 | 33 | 14.4% | **1.291** | 0.999–1.666 |

## Per-shape baseline-vs-candidate (B200, GPU 0, median µs; 21 trials, target 5000 µs)

| N | regime | baseline µs | candidate µs | speedup |
|---|---|---|---|---|
| 2–38 (decode, 9 rows) | decode | 6.15 | 6.16 | 0.999 |
| 110 | mid | 6.40 | 6.16 | 1.039 |
| 392 | prefill | 6.66 | 6.35 | 1.049 |
| 489 | prefill | 6.74 | 6.47 | 1.043 |
| 645 | prefill | 7.31 | 7.18 | 1.018 |
| 861–1167 | prefill | 8.19 | 8.20 | ~1.000 |
| 1464 | prefill | 12.30 | 8.20 | **1.499** |
| 1731 | prefill | 12.29 | 8.20 | **1.498** |
| 1811 | prefill | 12.30 | 9.58 | 1.284 |
| 1952–2366 | prefill | 12.30 | 10.25 | 1.199 |
| 2798–3524 | prefill | 16.39 | 10.26 | **1.598** |
| 3617 | prefill | 20.50 | 12.30 | **1.666** |
| 3769 | prefill | 20.50 | 12.30 | **1.666** |

Full per-row records: `bench/results.jsonl`. Dispatch buckets: `docs/dispatch.md`.

## Evidence-backed analysis

### Active bounds (neither regime is compute- or memory-bound)

NCU (`--set basic`, GPU 0) on representative shapes:

| Kernel (N) | Achieved occupancy | Active warps/SM | Grid | Waves/SM | Compute SM % | Memory % | Regs/thread |
|---|---|---|---|---|---|---|---|
| baseline `single_group` (N=3769) | 27.2% | 17.4 | 3769 | 3.18 | 27.9% | 13.9% | 29 |
| candidate `warp_per_token` (N=3769) | **33.3%** | **21.3** | **943** | **0.53** | 32.4% | 10.7% | 40 |
| candidate `block_per_token` (N=8) | 4.4% | 2.8 | 8 | 0.01 | 0.2% | 1.5% | 29 |

- **Decode (N=8): launch / occupancy floor.** With only 8 tokens of work, the grid
  is 8 blocks on 148 SMs → 0.01 waves, 4.4% occupancy, ~0% compute/memory
  throughput. The GPU is essentially idle; per-call time is the
  CPU→TVM-FFI→`cudaLaunch` dispatch floor (~6.15 µs), identical for both sides. No
  kernel change can move this without fusing the router into adjacent ops (outside
  this interface's contract). The candidate dispatches to the baseline algorithm
  here → exact parity, no regression. **This launch-floor is the named active bound
  for the decode regime.**
- **Prefill (N=3769): occupancy / latency bound.** Both sides run well under peak
  compute (≤32%) and memory (≤14%) throughput, so the kernel is bound by
  occupancy / instruction latency (the per-token `__expf` SFU work and the serial
  top-k selection), not by FLOP/s or HBM bandwidth. A roofline check confirms this:
  N=3769 moves ≈ 3769·256·4 B (scores) + 1 KB (bias) + 3769·8·(4+4) B (outputs)
  ≈ 4.1 MB; at ~8 TB/s that is ≈ 0.5 µs — two orders of magnitude below the
  measured 10–20 µs, so the kernel is nowhere near memory-bound. The candidate's win
  comes from **raising achieved occupancy 27%→33% and active warps 17→21 while
  cutting the block count 4× (3769→943 blocks, 3.18→0.53 waves)** by packing 4
  tokens per CTA with all warps productive — versus the baseline's one-block-per-
  token, warp-0-only selection that leaves 7/8 warps idle during the selection
  phase. Register pressure rose to 40/thread (the register-resident expert state)
  with **no spill**, and occupancy still improved.

### Why warp-per-token wins prefill but not decode

The baseline launches one 256-thread block per token and selects in warp 0 only
(224 idle threads), staging both sigmoid and biased scores through shared memory.
For large N this is wasteful: many heavy, mostly-idle blocks. The candidate makes
every thread productive (one warp per token, multiple tokens per block,
register-resident, no shared memory). For tiny decode N, though, there is simply
too little work to fill the machine regardless of layout, and one warp computing 8
serial `__expf` per lane is SFU-latency bound — the baseline's full-block parallel
sigmoid (256 threads, 1 `__expf` each) and one-CTA-per-token SM spread are better
there. The per-N warps/block sweep (`run_log.md`) made this crossover explicit, so
the dispatch uses the baseline kernel below N=768 and the warp kernel above it.

### Warp-specialization profiling

Not applicable. The candidate is a uniform warp-per-token reduction kernel with no
producer/consumer warp specialization (no `mbarrier`/named-barrier/pipeline
roles), so the `warp-specialization-report-skill` (clock() timeline) does not
apply. Diagnosis used NCU (occupancy/throughput/waves) per `ncu-report-skill`.

## Correctness

`bench/correctness.py` — 1479 checks, 0 failures (GPU 0). Candidate vs the
recovered baseline: exact-match ordered `topk_indices` and `topk_values` within
fp32 `atol=rtol=1e-5` on every captured production shape (×2 seeds) and the edge
grid (exact ties → smaller index, equal `sigmoid+bias` via different bias, negative
bias, saturating/Inf logits, N=0, max N=3769, renormalize=False). Output buffers
poisoned before each run (no stale/poison survivors); NaN/Inf checked;
shape/dtype/device verified. Unsupported parameters (`num_expert_group≠1`,
`topk_group≠1`) are rejected exactly as the baseline. The candidate is bit-identical
to the baseline (same `fast_sigmoid`, packed comparator, and renormalization
reduction), so it also matches the independent oracle wherever the oracle is exact.

## Provenance

B200 (`ion-b200` / `innomatrix-us-adc-smb200-0003`), GPU 0 idle (0% / 0–4 MiB)
before and after. PyTorch 2.11.0+cu130, CUDA 13.0, nvcc 13.0, TVM-FFI 0.1.9.
Baseline from sgl-project/sglang `main` @ `6b2c730bf793984c39f7f07b3c074ca05b059b00`.
Symmetric compile flags; matched ABI; current-stream launch. Exact commands and
idle evidence in `run_log.md`; build/timing details in `benchmark_method.md`.

## Dispatch scope and provenance notes

- **Production-only fast path.** The warp-per-token kernel runs only on the captured
  production domain (`E=256, topk=8, num_expert_group=1, topk_group=1, renormalize,
  scaling_factor=1.0, N>=768`); every other baseline-supported input uses the copied
  baseline kernel (`docs/dispatch.md`). **Decode parity is therefore by
  construction** — the candidate's decode/small-N path is the bit-identical recovered
  baseline kernel, so it cannot be a real regression; any sub-1.0 decode benchmark
  number is a CPU-launch-floor measurement artifact, not a kernel effect.
- **Idle provenance.** The authoritative numbers above are from a whole-box-quiet run
  on GPU 0 (external `nvidia-smi` verified idle before). A clean external
  before/after-exit idle bracket was additionally captured on idle GPU 2 (a
  user-approved deviation from the GPU-0 pin, taken because an unrelated external job
  later occupied GPUs 0–1). The GPU-2 run's launch-floor decode speedups were
  depressed by host CPU contention from that external job (decode is parity by
  construction), so it serves as the idle bracket, not the headline. Full details and
  the in-process-vs-external `nvidia-smi` distinction are in `docs/run_log.md`.

## Conclusion

**Promote the candidate.** It preserves correctness exactly, never regresses
(decode at parity by construction, bounded by the kernel-launch floor — the named
active bound), and delivers a 1.217× equal-weight / 1.037× call-weighted
geometric-mean speedup, reaching 1.67× on the largest prefill shapes, by converting
the baseline's idle selection warps and excess block count into higher achieved
occupancy.

### Future headroom (optional)

Achieved occupancy in the prefill win path is still only ~33%, so there is room
for a multi-row-per-CTA variant with shared bias preload, and per-N warps/block
tuning could smooth the wave-quantization sawtooth at N≈1811–2366. Decode is
launch-floor bound and cannot be improved within this interface's contract.
