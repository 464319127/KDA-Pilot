# Benchmark Method

## Kernel under test
`sgl_kernel.build_tree_kernel_efficient` (SGLang EAGLE/MTP speculative-decoding
tree builder). In place, returns `None`; mutates `tree_mask`, `positions`,
`retrive_index`, `retrive_next_token`, `retrive_next_sibling`. Captured GLM-5.2
B200 regime is a single fixed scalar point: `topk=1, depth=1, draft_token_num=2,
tree_mask_mode=FULL_MASK`, contiguous int64 inputs + bool `tree_mask`. Only `bs`
(1..10) and the bool `tree_mask` length `T` vary, with `T = 2*sum(verified_seq_len) + 4*bs`.

## ABI (baseline and candidate are symmetric)
Both sides are exposed through ONE torch CUDA extension
(`bench/csrc/binding.cpp`, built by `bench/build_ext.py`) that compiles
`baseline/build_tree_baseline.cu` and `solution/build_tree_candidate.cu`
together. Consequences:
- Identical registration/export style (pybind), identical build, identical
  compile flags, identical Python call path for both sides — so any wrapper /
  dispatch overhead is the same on both sides and cancels in the speedup ratio.
- Destination-passing: outputs are pre-allocated and passed in; the function
  returns `None` and mutates them in place (matches the captured op).
- Every launch uses `at::cuda::getCurrentCUDAStream()`.

We use the pybind torch-extension ABI (not TVM-FFI) for build reliability; it is
symmetric, which is the fairness requirement. The per-call host overhead is
identical on both sides; see "Measurement caveats".

## Compile flags (symmetric)
- `extra_cflags = ["-O3"]`, `extra_cuda_cflags = ["-O3"]` for the single combined
  build. No one-sided `--use_fast_math`; no architecture/math flags applied to one
  side only. The build is JIT (torch `cpp_extension.load`) and happens at import,
  outside any timed region.

## Timing harness
`bench/benchmark.py` is a verbatim copy of
`llm/docs/standalone_llm_benchmark_template.py` (unchanged timing logic):
- CUDA-event timing; warmup; one isolated subprocess per workload.
- Inner-loop amplification: each timed sample runs N back-to-back invocations
  inside one event pair, N grown until the sample reaches `target_sample_us=1000`
  or `inner_iterations_max=4096`. Essential for this launch-bound kernel.
- Interleaved A/B order per trial; `num_trials=7`, `warmup_runs=10`.
- Per workload: median / mean / std / min / p10 / p90; speedup =
  baseline_median / candidate_median; headline = equal-weight geomean of speedup
  over the `production: true` rows.
- Invocation:
  `CUDA_VISIBLE_DEVICES=6 python bench/benchmark.py --device cuda:0 --warmup-runs 10 --num-trials 7 --inner-iterations-min 1 --inner-iterations-max 4096 --target-sample-us 1000 --out bench/results.jsonl`

## Output pre-state + buffer ring (adapter)
The op depends on the FULL_MASK callsite pre-state: `tree_mask` prefilled `True`;
`retrive_next_token` / `retrive_next_sibling` prefilled `-1`. The harness poisons
`outputs` before the correctness call, so the adapter returns outputs as a custom
`RingOutputs` object (not a tensor/list/dict) — the harness' `_poison_outputs`
skips non-tensors, preserving the pre-state. Each invocation writes into the next
of `BUILD_TREE_RING` (default 512) pre-stated output sets so it never observes a
prior call's mutation; the correctness-gating compare always uses a freshly
pre-stated set. The op is data-independent (fixed per-request work; no loop over
`T`/`seq_len`; the only buffer-state-dependent path is one extra int64 store in
the baseline's `retrive_next_token` else-branch — a sub-nanosecond effect), so
timing is unbiased even if the ring wraps.

## Correctness (gate before any benchmark counts)
`bench/correctness.py`: EXACT match (int64 / bool / tree structure) of both
baseline and candidate against an INDEPENDENT pure-Python oracle derived from the
recovered semantics, plus candidate-vs-baseline; poison on the fully-written
buffers (`positions`, `retrive_index`) to catch partial writes; baseline and
candidate on SEPARATE copies (in-place); shape / dtype / device / stride checks;
multiple `verified_seq_len` distributions (uniform / skewed / monotonic / random);
a sweep over the full captured `(bs, T)` range (per-bs min/median/max T); and the
fallback rows (non-FULL_MASK mode, non-contiguous input) where the candidate must
reproduce the baseline bit-for-bit.

## Workload selection (headline vs regression)
`bench/workloads.json` is frozen from `docs/evidence.json` (see
`bench/gen_workloads.py`). The 183 distinct captured `(bs, T)` shapes collapse,
for the performance HEADLINE, to one representative production row per distinct
`bs` (1..10), each pinned to a real captured `(bs, T)` shape (median captured T
for that bs). This is justified — and explicitly documented, not silently dropped
— because the op's work and timing are determined by `bs` and are independent of
`T`; every captured bs bucket is represented, and correctness covers the full
`(bs, T)` range. Regression-only rows (`production: false`, excluded from the
headline) cover the extreme tree_mask lengths (T=36, T=11626), a degenerate
seq_len=0 case, and the baseline-fallback path. See `docs/dispatch.md`.

## Empty-kernel launch floor
`build_tree_noop` launches a do-nothing kernel with the same grid/block the
candidate fast path uses. Timed with the same CUDA-event + inner-loop method, it
gives the irreducible launch/scheduling latency on the target GPU — the lower
bound for any achievable op time and the key reference for the win/no-go verdict.

## Measurement caveats (launch-bound kernel)
This op writes only a few KB at most and runs as a single tiny launch on both
sides. The dominant cost is kernel launch / scheduling / per-call host overhead,
which is identical on both sides; therefore the headline ratio is expected to sit
near 1.0 and a kernel-body win is hard to surface through the standard per-call
harness. The empty-kernel floor quantifies this. A CUDA-graph-replay measurement
(DEC-3, auxiliary only) is used as a secondary diagnostic to isolate GPU-bound op
time when needed; it is not the headline.

## Provenance
See `docs/baseline_source.md` (upstream commit + copied files), `docs/run_log.md`
(host / GPU id / model + before/after idle), and `bench/results.jsonl`
(per-run records + harness provenance event with CUDA/torch versions).

<!-- Numbers (geomean, per-row medians, floor) are recorded in docs/results.md after the remote B200 run. -->
