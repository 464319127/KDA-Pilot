# Benchmark Method

## Kernel under test
`sgl_kernel.build_tree_kernel_efficient` (SGLang EAGLE/MTP tree builder). In place,
returns `None`; mutates `tree_mask`, `positions`, `retrive_index`,
`retrive_next_token`, `retrive_next_sibling`. Captured GLM-5.2 B200 regime: fixed
scalars `topk=1, depth=1, draft_token_num=2, tree_mask_mode=FULL_MASK`, contiguous
int64 inputs + bool `tree_mask`. Only `bs` (1..10) and the `tree_mask` length `T`
vary, with `T = 2*sum(verified_seq_len) + 4*bs`.

## ABI — local TVM-FFI direct-symbol (symmetric)
Baseline + candidate are compiled together in ONE module by `bench/build_ext.py`
via `tvm_ffi.cpp.load` (the repo's `diffusion/kernels/*` pattern), each host
launcher exported with `TVM_FFI_DLL_EXPORT_TYPED_FUNC` and taking
`tvm::ffi::TensorView` + `int64_t` scalars; the device `__global__` kernels are
unchanged. Destination-passing (outputs in place), launch on
`at::cuda::getCurrentCUDAStream()`. `bench/build_ext.py` exposes the only entry
points the harness uses — `baseline()`, `candidate()`, `noop()` — and torch tensors
are auto-marshalled to `TensorView` by tvm-ffi. Identical build / export / call
path for both sides (the fairness requirement).

## Compile flags (symmetric)
`extra_cflags=[-std=c++17,-O3]`, `extra_cuda_cflags=[-std=c++17,-O3,-gencode …sm_100]`,
torch ldlibs (`-lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda`). No one-sided
`--use_fast_math`. Build is JIT at import (outside any timed region).

## Timing harness
`bench/benchmark.py` is the verbatim repo template
(`standalone_llm_benchmark_template.py`): CUDA-event timing, warmup, one isolated
subprocess per workload, interleaved A/B, inner-loop amplification to
`target_sample_us=1000` (cap 4096), `num_trials=7`, `warmup_runs=10`; per workload
median/mean/std/min/p10/p90; speedup = baseline_median/candidate_median; headline =
equal-weight geomean over `production: true` rows. Invocation:
`CUDA_VISIBLE_DEVICES=6 python bench/benchmark.py --device cuda:0 --warmup-runs 10 --num-trials 7 --inner-iterations-min 1 --inner-iterations-max 4096 --target-sample-us 1000 --out bench/results.jsonl`.

## Workloads (AC-3: complete captured coverage)
`bench/workloads.json` is frozen from `docs/evidence.json` (`bench/gen_workloads.py`)
and contains **every distinct captured `(bs,T)` shape as a production row (183
rows)** — no captured bucket is dropped — each recording shapes, dtypes, strides
(all contiguous; `is_contiguous=True` in the evidence), the fixed scalars, exact
tolerance (atol=rtol=0), and a seed. Plus 4 regression-only rows (`production:false`,
excluded from the headline): two degenerate seq=0 edges, a QLEN_ONLY fallback, and
a non-contiguous fallback. (Off-shape guard conditions — draft≠2, wrong
`parent_list` dtype, wrong `selected_index` shape — are exercised directly in
`bench/correctness.py`.)

## Output pre-state + no-wrap buffer ring (AC-6)
The op depends on the FULL_MASK callsite pre-state (`tree_mask`=True;
`retrive_next_token`/`retrive_next_sibling`=-1). The harness poisons `outputs`
before the correctness call, so the adapter returns a custom `RingOutputs` object
(not a tensor/list/dict) — `_poison_outputs` skips it, preserving the pre-state.
The ring holds 5 contiguous 2D tensors (`RING × elems`) whose rows are the per-call
output sets; `RING=16384` exceeds the maximum per-trial invocation count for the
configured benchmark (correctness 1 + warmup + calibration sum ≤8191 + timed
≤inner_max=4096 ≈ 12298), and `RingOutputs.next()` **hard-asserts it never wraps**,
so every warmup/calibration/timed invocation writes into a fresh, pre-stated set
and never observes a prior call's mutation. make_case is called per trial, so the
cursor resets each trial. The 2D layout keeps memory + allocation cheap.

## Correctness (gate before any benchmark)
`bench/correctness.py`: EXACT int/bool/tree match of baseline AND candidate vs an
independent Python oracle; poison on the fully-written outputs; separate in-place
copies; shape/dtype/device/stride checks; 4 distributions; full captured `(bs,T)`
sweep; 5 fallback cases. 691 cases, 0 failures on B200.

## Route coverage proof
A TVM-FFI diagnostic `build_tree_candidate_route` (exported alongside the ops;
`build_ext.route`) runs the SAME predicate as the candidate dispatcher
(`candidate_fast_path_eligible`) and returns 1=fast path / 0=baseline fallback
without launching. `bench/correctness.py` ASSERTS route==1 for every captured
production row (all bs 1..10) and route==0 for the 5 off-domain cases — proving the
native kernel is actually exercised over the captured regime (not silently falling
back). This is required because output-equality alone passes even when the candidate
wrongly falls back to the baseline (the Round-1 `is_contiguous` zero-size-dim bug).
The contiguity helper now treats any zero-element tensor (e.g. `parent_list [bs,0]`)
as contiguous, matching PyTorch.

## Empty-kernel floor + controlled probe + wrapper diagnostic
`bench/floor_probe.py`: (A) controlled same-process A/B probe for ALL bs 1–10
(fresh retrieve buffers per call via a ring → no baseline reuse artifact), 31
trials, tight p10/p90, vs the empty-kernel (`build_tree_noop`, same grid) launch
floor; (B) the MEASURED wrapper-inclusive path (alloc + `tree_mask.fill_(True)` +
`retrieve_buf=full(-1)` + op) vs op-only. The controlled probe removes the
cross-subprocess GPU-clock noise of the official harness and is the clock-noise-free
verdict.

## Active bound (why this is a no-go)
The op writes <a few KB and runs as a single tiny launch. Per-call cost (Round-2,
candidate genuinely running) is dominated by the empty-kernel launch floor
(~4.4–6.7µs) plus multi-arg TVM-FFI marshalling; baseline and candidate both land
~10.6–12.6µs, so the kernel body is a sub-µs sliver with no headroom. No
kernel-body design can produce a stable win under the strict op ABI — the
candidate's single-block layout is even marginally worse than the baseline's
bs-block parallel launch for bs>1 (mild regression, geomean 0.984). The only
material lever is the wrapper prefill (~64% of the realistic path, measured),
which DEC-1 holds out of promotion scope.

## Provenance
`docs/results.md` (numbers + verdict), `docs/dispatch.md` (dispatch table),
`docs/run_log.md` (host/GPU/idle/versions/commands), `bench/results.jsonl` (raw,
gitignored), `bench/floor_probe_out.txt` (controlled probe + wrapper).
