# Benchmark Method — mega_kernel task 01 (mnnvl_ar_jit_bs1)

Authoritative measurement: `bench/ar_harness.py` implementing config.toml's
`[benchmark]` mode — **"8x per-device 50-round CUDA graphs, concurrent replay,
wall/round"** (pattern: `prior/ar_bench.py`), on the shared single-process NVLS
workspace (`bench/sp_nvls_workspace.py`), for the frozen rows in
`bench/workloads.json` (T in {1,6}, H=6144, bf16, world=8, eps=1e-5).

## Documented deviations from the single-GPU llm/docs wording

The repo-wide contracts are written for single-GPU single-tensor kernels. This
task's kernel is a world=8 NVLS/cuMulticast collective; four clauses are
structurally inapplicable and are overridden by the task contract
(`prompt.md`, `config.toml`, `SHAPES.md`), whose precedence the draft's own
escape clauses establish. Endorsed by the Codex first-pass review.

- **D1 — All-8-GPU validation instead of one idle GPU.**
  Conflicting clause (llm/docs/standalone_llm_benchmark.md, Hard Rules): "Use
  exactly one idle GPU of the task's target architecture for correctness,
  benchmark, profiler, and NCU commands." A world=8 multicast collective
  cannot execute on one GPU. Adaptation: the idleness policy applies to all 8
  GPUs — `nvidia-smi` (util + memory) for all 8 recorded before/after every
  measurement (harness does this automatically per JSONL record); performance
  data valid only when the 8 cards run nothing but the task process. The
  resident GLM-5.2 serving instance holds memory but idles at 0% util; the
  measured noise floor (below) quantifies whether request-idle coexistence is
  acceptable; if not, serving is stopped for final numbers and restarted via
  the documented command.
- **D2 — Serving gates are acceptance gates, not provenance.**
  Conflicting clause: "Do not start `sglang serve` ... or require a
  TP/EP/multi-GPU deployment for task validation." The source prompt
  explicitly defines serving gates (P0.c sanity >= 376 flag-ON; promote
  >= 378 + official 3x40) and provides the restart command — the boundary
  clause's own exception ("unless the user explicitly asks for a separate
  serving validation pass") applies. Serving passes run only at milestone
  boundaries; per-iteration validation is this standalone harness.
- **D3 — Timing core.**
  Conflicting clause: "`bench/benchmark.py` must start from
  `standalone_llm_benchmark_template.py`. Do not invent a different timing
  harness unless this template has a documented bug..." The template is copied
  verbatim (sha256 2e1712e5..ee13 both sides) for contract compliance, but its
  single-device CUDA-event loop cannot measure a spin/flag collective: an
  event pair on one rank times that rank's spin, not the collective, and the
  per-workload single-`Process` isolation cannot host the single-process
  8-GPU workspace config.toml requires. `bench/adapter.py` therefore refuses
  with this explanation (a wrong measurement is worse than no measurement),
  and `bench/ar_harness.py` is authoritative while preserving every
  transferable template policy: frozen workloads, fresh inputs per trial,
  randomized A/B order per trial, output poisoning, median/mean/std/min/
  p10/p90 per side, speedup = baseline_median/candidate_median, no-silent-skip
  (empty row selection or any failed row aborts), JSONL records with full
  provenance.
- **D4 — Baseline identity.** See `docs/baseline_source.md`: the baseline is
  flashinfer 0.6.12's header (sglang is the port destination; "latest upstream
  SGLang main" resolution is replaced by flashinfer provenance + the sglang
  destination pin `87992eeec`).

## Timing method (config.toml convention)

- Per device: one CUDA graph containing 50 back-to-back fused-AR launches on
  an explicit per-device stream (explicit stream required for capture —
  campaign lesson).
- Timed sample: `reps=10` x (replay all 8 graphs concurrently, then sync all
  8); wall_us_per_round = elapsed / (reps * 50). This is the same convention
  behind the campaign's 8.3us serving figure and `prior/ar_bench.py`.
- Trials: 7 per row per implementation; fresh inputs each trial
  (seed = base + trial * 1_000_003 + T); implementation order randomized per
  trial (interleaved A/B); 3 warmup replays before timing; one eager warmup
  round before capture (JIT/op registration, allocator), workspace reset to
  the canonical Lamport state before every capture so both sides start
  identically.
- Workspace: one multicast object bound on all 8 devices + per-device unicast
  buffers, 3 Lamport buffers, flags `[cur=0, dirty=2, bytesPerBuffer, 0,
  0,0,0,0, counter=0]`, buffers pre-filled with the fp32 -0.0 bit pattern —
  KERNEL-ABI FAITHFUL to `MNNVLAllReduceFusionWorkspace` (everything the
  kernel reads is identical), while the allocation mechanism differs by
  design: single-process VMM/cuMulticast with MINIMUM multicast granularity
  and all-device UC access, versus flashinfer's per-process torch symmetric
  memory with signal pad. Both implementations run on the SAME workspace
  instance, so the allocation difference cannot bias the A/B.
- PDL: `--pdl` applies to BOTH sides identically (default 0 for primary
  numbers: back-to-back AR rounds without intervening kernels would let PDL
  overlap consecutive AR calls, which does not represent the serving graph
  where other kernels sit between AR calls; PDL evaluation for P1 happens
  with a predecessor-aware setup per the plan).

## Correctness method

- Primary gate: bf16 bitwise equality (int16 bit-cast compare) of `out` AND
  `residual_out`, port vs flashinfer original, every rank, T in {1,6}, same
  inputs, same workspace protocol, after identical round counts.
- Oracle sanity: composed fp32 reference (fp32 cross-rank sum + fp32 residual
  add + fp32-accumulation rmsnorm), elementwise atol 7e-2 / rtol 2e-2 (the
  correctness contract's bf16 row). Measured fact (smoke_fi, 2026-07-10): the
  flashinfer ORIGINAL vs this oracle shows max-rel 3.904e-3 = one bf16 ulp
  (2^-8) — the irreducible quantization floor. The source prompt's fallback
  wording "fp32 oracle rel<1e-3" is unsatisfiable under a max-rel metric for
  ANY bf16 kernel including the baseline itself; the contract tolerance is
  the operative oracle gate, the 1e-3 figure is kept as a diagnostic
  reference, and the bitwise A/B gate (which the prompt makes primary) is
  unaffected. Recorded in the goal tracker's Plan Evolution Log.
- Structural checks per output: shape/dtype/stride vs reference, NaN/Inf,
  poisoned-output detection (outputs NaN-filled before every run).
- Comparator self-tests (must-fail cases): a flipped bf16 bit must fail the
  bitwise comparator; a wrong eps (1e-6) must fail the oracle in the
  small-magnitude regime where eps dominates; poisoned outputs must be caught
  by the structural check. `bench/ar_harness.py --mode selftest`.
- Race detector: `--mode stability` (review-hardened design) replays an
  instrumented graph where every round (a) copies one of THREE rotating input
  banks into the live input tensors — so a stale read from a previous Lamport
  epoch carries different values and cannot compare bitwise-equal (three
  banks cover 1- and 2-epoch staleness, the ring depth), (b) NaN-poisons the
  outputs in-graph — so a silently skipped round cannot equal any reference,
  (c) launches, (d) accumulates bitwise mismatches vs the per-bank reference
  in-graph. 1000 replays x 50 rounds = 50,000 distinguishable-data checks
  per row. A plain uninstrumented graph (production-shaped, maximum
  cross-round overlap) additionally runs 1000 replays with a final-output
  check per replay. Both must be clean. (The first-cut detector used constant
  inputs and reused output buffers, which could miss stale-epoch reads and
  no-op rounds — flagged by the independent method review and fixed before
  any stability gate is claimed.)

## Compile-flag symmetry

- Baseline side: the AOT wheel binary `flashinfer-jit-cache 0.6.12+cu130`
  (what the serving stack actually loads; flashinfer's loader prefers it over
  local JIT). SASS inspection shows it is a FAST-MATH build (`.FTZ`
  modifiers throughout, `MUFU.RCP` approximate division).
- Candidate side: TVM-FFI build, `TVM_FFI_CUDA_ARCH_LIST` pinned to 10.3a,
  `-std=c++20 -O3 --expt-relaxed-constexpr --use_fast_math
  -DSGL_CUDA_ARCH=1030`. `--use_fast_math` MATCHES the baseline binary's
  float semantics — required for bf16 bit-exactness on subnormal/zero-sign
  value classes (see docs/results.md "verbatim source is not enough"); this
  is flag SYMMETRY with the deployed baseline, not a one-sided fast-math
  advantage. Verified by `bench/value_zoo_ab.py` (adversarial value classes)
  in addition to randn rows.
- Launch path symmetry: both sides drive the identical FFI signature
  (caller-allocated outputs, per-device current stream, same workspace
  pointers/flags); neither side goes through the live serving checkout.

## Noise floor

- `--mode noise` runs the full A/B protocol with the SAME implementation on
  both slots (fi vs fi): the |1 - pseudo_speedup| spread and per-side std are
  the measured noise floor used for "beyond noise" promotion judgments
  (AC-7.1) and for the +-3% parity-gate margin. Result recorded in
  `bench/results.jsonl` and `docs/results.md`.
