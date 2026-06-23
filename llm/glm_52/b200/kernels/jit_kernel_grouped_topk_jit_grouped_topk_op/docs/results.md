# Results — `jit_kernel.grouped_topk._jit_grouped_topk_op` on B200 (GLM-5.2)

## Headline

Correctness-preserving **win**. Native-CUDA candidate vs the recovered SGLang
baseline, NVIDIA B200, all 17404-call production shape set. Authoritative benchmark
(`bench/results.jsonl`) on **idle GPU 6** (external `nvidia-smi` before AND
after-exit both `0% / 0 MiB`; a recorded, user-approved plan revision from the
`REMOTE_GPU_ID=0` pin because an external job occupied GPU 0 — see `run_log.md`).
All numbers below are **measured** (no row substitution).

- **Equal-weight geometric-mean speedup over the 43 production rows: 1.220×**
- **Call-weighted geometric mean (by captured call frequency): 1.038×**
- **No regression on any production row** (min 0.999×, i.e. parity at the noise floor).
- Range: decode/small-N parity (1.000×) → prefill up to **1.67×**.
- Correctness: full grid **1693 checks**, 0 failures, vs baseline + an independent oracle.

The candidate dispatches by token count: the recovered baseline kernel for the
launch-floor-bound decode/small regime (parity), a register-resident
**warp-per-token** kernel for large-prefill (the win). Decode measures at parity
because that path *is* the copied baseline kernel.

## Per-regime summary

| Regime | N | rows | % of calls | geomean speedup | range |
|---|---|---|---|---|---|
| decode | 2–38 | 9 | **85.2%** | **1.000** (parity) | 1.000–1.000 |
| mid | 110 | 1 | 0.4% | 1.009 | — |
| prefill | 392–3769 | 33 | 14.4% | **1.296** | 0.999–1.674 |

## Per-shape (idle GPU 6, authoritative; median µs; 21 trials, target 5000 µs, --no-isolated)

| N | regime | baseline µs | candidate µs | speedup |
|---|---|---|---|---|
| 2–38 (decode, 9 rows) | decode | 6.16 | 6.16 | **1.000** |
| 110 | mid | 6.21 | 6.16 | 1.009 |
| 392 | prefill | 6.66 | 6.16 | 1.081 |
| 489 | prefill | 6.66 | 6.41 | 1.040 |
| 645 | prefill | 7.18 | 7.18 | 1.000 |
| 861 | prefill | 8.19 | 8.20 | 0.999 |
| 1167 | prefill | 8.20 | 8.20 | 1.000 |
| 1464 | prefill | 12.30 | 8.21 | **1.499** |
| 1731 | prefill | 12.30 | 8.20 | **1.499** |
| 1811 | prefill | 12.30 | 9.23 | 1.333 |
| 2247 | prefill | 12.30 | 10.25 | 1.200 |
| 2798 | prefill | 16.39 | 10.26 | **1.598** |
| 3617 | prefill | 20.51 | 12.25 | **1.674** |
| 3769 | prefill | 20.51 | 12.30 | 1.667 |

Full per-row records: `bench/results.jsonl` (idle GPU 6). Dispatch: `docs/dispatch.md`.
The decode rows measure exactly at parity (the candidate runs the copied baseline
kernel there); the large-prefill win (N≥1464: 1.33–1.67×) is the contribution. This
run reproduces the round-0 quiet-box GPU-0 numbers (1.217× / 1.037×), confirming it.

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
the baseline. The decode/small-N + off-domain **fallback path is the copied baseline
kernel → bit-identical by construction**. The large-N **warp path** reuses the same
selection math (`fast_sigmoid`, packed comparator, renormalization reduction) and is
validated as **exact ordered `topk_indices` + `topk_values` within fp32 tolerance**
against both the baseline and the independent oracle (the recorded check is
tolerance-based, not a bytewise comparison).

## Measurement fairness (root-cause of earlier rounds' apparent decode regression)

Earlier benchmark runs showed a spurious decode regression (≈0.78–0.83) whenever
any external job ran elsewhere on the box, even on an idle target GPU — yet a
controlled direct micro-benchmark always showed decode at exactly 1.000. The cause
was an **asymmetric per-call filesystem `stat()` in the adapter**: `call_candidate`
invoked `has_candidate()` (a `Path.is_file()` stat) on every invocation, while
`call_baseline` did not. For this CPU-launch-bound (~6 µs) kernel that stat is
negligible with a hot dentry cache (quiet box → decode 0.999) but balloons under
host/IO load, biasing only the candidate side. The fix resolves candidate
availability **once at import** so both call paths do only a cached module lookup +
launch (`bench/adapter.py`). After the fix, the authoritative idle-GPU-6 run measures
**decode = 1.000** even with the external job present — i.e. the decode rows now
*measure* at parity (not by substitution). Captured as BitLesson
`BL-20260623-adapter-percall-stat`.

## Provenance

B200 (`ion-b200` / `innomatrix-us-adc-smb200-0003`). **Authoritative benchmark on
idle GPU 6** (external `nvidia-smi` BEFORE and AFTER-exit both `0% / 0 MiB`) — a
recorded, user-approved plan revision from the `REMOTE_GPU_ID=0` pin because an
external job occupied GPU 0. The round-0 quiet-box GPU-0 run (1.217× / 1.037×)
independently reproduces these numbers. PyTorch 2.11.0+cu130, CUDA 13.0, nvcc 13.0,
TVM-FFI 0.1.9. Baseline from sgl-project/sglang `main` @
`6b2c730bf793984c39f7f07b3c074ca05b059b00`. Symmetric compile flags; matched ABI;
current-stream launch. The in-process `nvidia_smi_after` inside `results.jsonl` is a
diagnostic (the benchmark's own resident context), not the idle check. Exact commands
+ idle evidence: `run_log.md`; build/timing: `benchmark_method.md`.

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

**Promote the candidate.** It preserves correctness exactly, never regresses (decode
**measures** at parity — that path is the copied baseline kernel — bounded by the
kernel-launch floor, the named active bound), and delivers a **measured 1.220×
equal-weight / 1.038× call-weighted** geometric-mean speedup, reaching 1.67× on the
largest prefill shapes, by converting the baseline's idle selection warps and excess
block count into higher achieved occupancy. (Round-0 quiet-box GPU-0 independently
reproduced 1.217× / 1.037×.)

### Future headroom (optional)

Achieved occupancy in the prefill win path is still only ~33%, so there is room
for a multi-row-per-CTA variant with shared bias preload, and per-N warps/block
tuning could smooth the wave-quantization sawtooth at N≈1811–2366. Decode is
launch-floor bound and cannot be improved within this interface's contract.
