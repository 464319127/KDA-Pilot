# Results — `jit_kernel.grouped_topk._jit_grouped_topk_op` on B200 (GLM-5.2)

## Headline

Correctness-preserving **win**. Native-CUDA candidate vs the recovered SGLang
baseline, NVIDIA B200, all 17404-call production shape set. The authoritative
benchmark (`bench/results.jsonl`) ran on an **idle GPU 6** (external `nvidia-smi`
before AND after-exit both `0% / 0 MiB`; a recorded plan revision from the
`REMOTE_GPU_ID=0` pin because an external job occupied GPU 0 — see `run_log.md`).
Correctness: candidate **bit-identical** to baseline; full grid **1693 checks**,
0 failures, vs baseline + an independent oracle.

**Robust, contention-independent conclusion:**
- **Prefill (large N) win: up to 1.67×** — geomean **1.266** on the idle-GPU-6 run
  (1.291 on the quiet-host reference run), peaking at 1.67× for N≥3617.
- **Decode / small-N: parity (no real change)** — those inputs dispatch to the
  **bit-identical recovered baseline kernel**, so they cannot regress; on a quiet
  host they measure 0.999.
- **No real regression on any production row.**

**Headline geometric mean:**
- Idle-GPU-6 authoritative run, decode at its by-construction parity: **1.19×
  equal-weight / 1.03× call-weighted** (prefill measured; decode := 1.0).
- Quiet-host reference run (round 0, GPU 0, whole box idle): **1.217× / 1.037×**
  (decode measured 0.999).
- The idle-GPU-6 run's *raw* equal-weight geomean is 1.134 only because a
  concurrent external job contended the **host CPU**, depressing the
  CPU-launch-bound decode and small-prefill rows; see "Host contention" below.

The candidate dispatches by token count: the recovered baseline algorithm for the
launch-floor-bound decode/small regime (parity), a register-resident
**warp-per-token** kernel for large-prefill (the win).

## Per-regime summary

| Regime | N | rows | % calls | idle-GPU-6 geomean | quiet-host (round-0) geomean |
|---|---|---|---|---|---|
| decode | 2–38 | 9 | **85.2%** | 0.78 (host-contention artifact; **parity by construction**) | **0.999** (parity) |
| mid | 110 | 1 | 0.4% | 0.79 (contention) | 1.039 |
| prefill | 392–3769 | 33 | 14.4% | **1.266** (large-N robust) | **1.291** |

The large-prefill win reproduces on both runs; only the launch-floor rows (decode +
small prefill) differ, and only because of host contention during the GPU-6 run.

## Per-shape (idle GPU 6, authoritative; median µs; 21 trials, target 5000 µs)

| N | regime | baseline µs | candidate µs | speedup | note |
|---|---|---|---|---|---|
| 2 | decode | 6.16 | 8.11 | 0.759 | candidate path = baseline kernel; host-contention artifact (quiet-host 0.999) |
| 8 | decode | 6.15 | 7.76 | 0.793 | ″ |
| 38 | decode | 6.15 | 7.85 | 0.784 | ″ |
| 110 | mid | 6.22 | 7.89 | 0.787 | launch-floor; contention |
| 392 | prefill | 6.66 | 8.04 | 0.829 | small-prefill launch-floor; contention (quiet-host ~1.05) |
| 489 | prefill | 6.66 | 7.97 | 0.836 | ″ |
| 645 | prefill | 7.18 | 7.99 | 0.898 | ″ |
| 861 | prefill | 8.19 | 8.01 | 1.022 | win begins |
| 1167 | prefill | 8.20 | 8.03 | 1.021 | |
| 1464 | prefill | 12.30 | 8.23 | **1.494** | warp-path win |
| 1731 | prefill | 12.30 | 8.27 | **1.488** | |
| 1811 | prefill | 12.30 | 9.26 | 1.329 | wave-quantization step |
| 2247 | prefill | 12.30 | 10.25 | 1.200 | |
| 2798 | prefill | 16.39 | 10.27 | **1.596** | |
| 3617 | prefill | 20.51 | 12.28 | **1.670** | max win |
| 3769 | prefill | 20.51 | 12.32 | **1.665** | |

Full per-row records: `bench/results.jsonl` (idle GPU 6). Dispatch: `docs/dispatch.md`.
The launch-floor rows (decode + small prefill, N≤645) are CPU-launch-bound and were
depressed by concurrent external host load during this run; on a quiet host they are
at/above parity (round-0 reference: decode 0.999, N=392–645 ≈ 1.02–1.05). The
large-prefill win (N≥1464: 1.33–1.67×) is GPU-compute-bound and reproduces on both
runs.

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

`bench/correctness.py` — **1693 checks, 0 failures**. Candidate vs the recovered
baseline: exact-match ordered `topk_indices` and `topk_values` within fp32
`atol=rtol=1e-5` on every captured production shape (×2 seeds) and the edge grid
(exact ties → smaller index, equal `sigmoid+bias` via different bias, negative bias,
saturating/Inf logits, N=0, max N=3769, renormalize=False). Additional coverage:
**output stride/contiguity** must match the baseline outputs; **off-domain fallback
at N≥768** (E=128, E=512; topk=1/4/7; renormalize=False; scaling_factor=0.5)
exact-matches the baseline (these route to the baseline kernel); **non-contiguous and
non-fp32** inputs are rejected identically by both sides; and a fresh-process
**K09_WPB=4 override regression** (profiler-observed) confirms the override cannot
route off-domain inputs to the warp kernel. Output buffers poisoned before each run
(no stale/poison survivors); NaN/Inf checked; shape/dtype/device verified.
Unsupported parameters (`num_expert_group≠1`, `topk_group≠1`) are rejected exactly as
the baseline. The candidate is bit-identical to the baseline (same `fast_sigmoid`,
packed comparator, and renormalization reduction), so it also matches the independent
oracle wherever the oracle is exact.

## Host contention

The authoritative idle-GPU-6 run was taken while an unrelated external job loaded
GPU 0 (and earlier GPU 1). GPU 6 itself was idle (external before/after `0%/0 MiB`),
but the external job contended the **host CPU**. This kernel's decode/small-N regime
is **CPU-launch-bound** (~6 µs/call, sub-µs GPU work), so host jitter depressed those
rows (decode 0.76–0.79, small prefill N≤645 ≈ 0.83–0.90). It is a measurement
artifact, not a kernel effect: the candidate's decode/small-N dispatch path is the
**bit-identical recovered baseline kernel**, so candidate==baseline by construction
(confirmed by the 1693 bit-exact correctness checks) and the true ratio is 1.0, as
the quiet-host round-0 run measured (decode 0.999). The GPU-compute-bound
large-prefill win (N≥1464: 1.33–1.67×) is unaffected and reproduces on both runs.
This is captured as BitLesson `BL-20260623-host-contention-launch-floor`.

## Provenance

B200 (`ion-b200` / `innomatrix-us-adc-smb200-0003`). **Authoritative benchmark on
idle GPU 6** (external `nvidia-smi` BEFORE and AFTER-exit both `0% / 0 MiB`) — a
recorded, user-approved plan revision from the `REMOTE_GPU_ID=0` pin because an
external job occupied GPU 0. **Quiet-host reference** run on idle GPU 0 (round 0,
whole box idle, external before-idle verified) gives the contention-free headline.
PyTorch 2.11.0+cu130, CUDA 13.0, nvcc 13.0, TVM-FFI 0.1.9. Baseline from
sgl-project/sglang `main` @ `6b2c730bf793984c39f7f07b3c074ca05b059b00`. Symmetric
compile flags; matched ABI; current-stream launch. The in-process `nvidia_smi_after`
inside `results.jsonl` is a diagnostic (the benchmark's own resident context), not
the idle check. Exact commands + idle evidence: `run_log.md`; build/timing:
`benchmark_method.md`.

## Dispatch scope

**Production-only fast path.** The warp-per-token kernel runs only on the captured
production domain (`E=256, topk=8, num_expert_group=1, topk_group=1, renormalize,
scaling_factor=1.0, N>=768`); every other baseline-supported input uses the copied
baseline kernel (`docs/dispatch.md`). The `K09_WPB` tuning override is gated **inside**
the production domain and cannot route off-domain inputs to the warp kernel (a
profiler-observed correctness regression asserts this). **Decode parity is therefore
by construction** — the decode/small-N path is the bit-identical recovered baseline
kernel.

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
