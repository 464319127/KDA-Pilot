# Remote Run Log — fast_topk_transform_fused (B200)

All GPU work runs on the task's pinned host/GPU. No fabricated evidence; entries are appended per session.

## Session — Round 2 (ABI build + differential probe)

- Host: `ion-b200` → `innomatrix-us-adc-smb200-0003`, user `sglang-omni`, container `sglang_bbuf` (Up 8 days).
- Toolchain: torch `2.11.0+cu130`, CUDA `13.0`, nvcc `13.0.88`, `tvm_ffi 0.1.9`, flashinfer present.
- Task workspace on remote: `/home/sglang-omni/bbuf/kda_topk` (synced from the local kernel folder via tar; `.humanize`/`__pycache__` excluded).
- Pinned GPU: id 1 (`REMOTE_GPU_ID=1` → `CUDA_VISIBLE_DEVICES=1`).

### GPU state (nvidia-smi, before work)
```
0, NVIDIA B200, util 5%,  mem 53638/183359 MiB
1, NVIDIA B200, util 5%,  mem 41472/183359 MiB   <-- pinned GPU 1: NOT idle (other tenant ~41 GB)
2, NVIDIA B200, util 8%,  mem 85054/183359 MiB
3..7 idle (0 util, 0 mem except 5=156764)
```
GPU 1 is **not idle** this session. Correctness/probe (exact-match, not timing-sensitive) ran on GPU 1's ~142 GB free memory — valid for correctness. **Timing/benchmark (AC-7) is deferred** until GPU 1 is idle (verified before+after), per the task policy; no benchmark numbers were taken this session.

### Commands
```
# build the TVM-FFI ABI (cached in solution/)
cd /home/sglang-omni/bbuf/kda_topk && TORCH_CUDA_ARCH_LIST=10.0 python3 solution/build.py
# -> built+loaded topk_transform_abi : OK

# differential probe (one output + naive-oracle correctness + candidate==baseline)
CUDA_VISIBLE_DEVICES=1 python3 solution/_probe.py
```

### Results
- **ABI build:** SUCCESS. `tvm_ffi.cpp.load_inline` compiled baseline `topk.cu` + candidate + TVM-FFI binding into one `topk_transform_abi` module (sm_100, `-std=c++17`, no fast-math; torch include/lib linked for the ATen-based baseline). Module type `tvm_ffi.module.Module`; functions accessed via attribute (`mod.fast_topk_transform_fused_baseline(...)`).
- **Output count:** confirmed ONE `(B, topk)` int32 output (destination-passing signature; the op writes exactly `dst_out`). Final hardware confirmation of the static finding in `docs/baseline_source.md`.
- **Naive path (`length <= topk`)** — `baseline == naive oracle` AND `candidate == baseline`, exact, on: decode (B=2,N=64,M=40), N==topk (B=8,N=2048), ties (B=8,N=256), prefill (B=16,N=448,S=4). This is the dominant production regime (3674/4246 captured calls have `N < topk`).
- **Radix path (`length > topk`, e.g. B=8,N=2112)** — `candidate != baseline` **even though the candidate forwards to the identical baseline code**. The baseline `fast_topk_cuda_tl` is therefore **non-deterministic**: it quantizes each score to an 8-bit key (`convert_to_uint8`), so many distinct float scores collide into the boundary histogram bucket, and the fill of the last `k` slots from that bucket is thread/race-ordered. Two runs on identical inputs select different (equally-valid) indices.

### Consequence (carried to next round)
AC-4's literal "exact-match selected indices + baseline-identical tie-break" is **infeasible for the radix path** because the baseline itself is non-deterministic there. The correctness criterion must split by regime: naive path = exact-match (works now); radix path = a tie-tolerant valid-top-k criterion (selected indices' 8-bit keys all ≥ the threshold key; per-bucket counts and the transform/pad match), comparing against the baseline's 8-bit-key selection semantics rather than a single non-deterministic run.

## Session — Round 3 (regime-split correctness gate → matched_ratio == 1.0)

- Same host/container/toolchain; pinned GPU 1 (`CUDA_VISIBLE_DEVICES=1`); GPU 1 still **not idle** (other tenant). Correctness only (not timing-sensitive); **no benchmark timing taken** (AC-7 timing still deferred until GPU 1 idle).
- Refined the radix criterion from the Round-2 sketch: `fast_topk_cuda_tl` does an 8-bit coarse histogram then **refines the boundary bin via full 32-bit `convert_to_uint32` radix rounds** (topk.cu ~184-251), so it selects the TRUE top-k by full float precision; only output ORDER (and exactly-equal-value ties) is race-nondeterministic. So the radix criterion is a full-float VALID-top-k check, not an 8-bit bucket check.

### Commands + results
```
CUDA_VISIBLE_DEVICES=1 python3 bench/correctness.py cuda:0
-> matched_ratio = 1.0000  (248/248 workloads)   [236 production + 12 regression]

CUDA_VISIBLE_DEVICES=1 python3 solution/_probe.py
-> naive cases: candidate==baseline==oracle (exact)
-> radix_gt_topk (B=8,N=2112): baseline_valid_topk=True candidate_valid_topk=True
   (exact_order_match=False, expected) ; PROBE_OK ; exit 0
```
- **AC-4 gate PASSES**: `bench/correctness.py` regime-splits — naive (`length<=topk`) exact candidate==baseline==oracle; radix (`length>topk`) validates EACH output as a valid top-k by full float key (recover selected positions from transformed entries, require distinct & in `[0,length)` & count topk, selected score multiset == `torch.topk(score[:length], topk)` values). Output poisoning + shape/dtype/device/contiguity/finite checks retained.
- Build was cached (no rebuild needed this session).

## Session — Round 4 (captured-contract coverage: row_starts + non-linear page table)

- Same host/container/toolchain; pinned GPU 1; still **not idle** → correctness only, **no timing** (AC-7 deferred).
- Closed the R3-review AC-3/AC-4 coverage gaps: (1) the 4 large-prefill production variants pass a `(B,)` int32 `row_starts` tensor (verified in `docs/evidence.json`) — `gen_workloads.py` now keys `row_starts_kind`, the adapter synthesizes valid `row_starts` (`row_start+length<=N`), and `validate_topk` reads the ragged window `score[b, row_start:row_start+length]`; (2) `validate_topk` now inverts each output entry through the ACTUAL `page_table[seq]` row (per-sequence scatter inverse) instead of the linear `out - page_table[s,0]` shortcut, and a non-linear (permuted) page-table regression row was added.

### Command + result
```
CUDA_VISIBLE_DEVICES=1 python3 bench/correctness.py cuda:0
-> matched_ratio = 1.0000  (251/251 workloads)
   [236 production (incl. 4 row_starts=tensor large-prefill) + 15 regression
    (incl. permuted_naive, permuted_radix, row_starts_radix)]
```
- **AC-3/AC-4 captured-contract coverage now complete**: the grid exercises `row_starts` tensor rows, non-linear page-table transform, naive exact-match, and radix valid-top-k. Build cached.

## Session — Round 6 (freeze immutable baseline numbers)

- Same host/container/toolchain; pinned GPU 1 (`CUDA_VISIBLE_DEVICES=1`, `--device cuda:0`).
- **GPU-1 idle evidence (AC-7):** before `index 1: util 0%, mem 59666 MiB`; after `index 1: util 0%,
  mem 59666 MiB`. A parked process (pid 3048677, sharded GPU0/1) holds 59 GB at **util 0%** (no active
  compute) and never woke during the run; no cleanly-idle GPU existed (3/4/5/6/7 were 149-157 GB or
  util 7-73%). `util 0%` before+after → timing valid on the pinned GPU; the 59 GB resident allocation
  does not contend for SMs/bandwidth (documented caveat).
- Two harness bugs fixed to make the template benchmark runnable (commit this round):
  (1) `bench/adapter.py` `Case` changed from `@dataclass` to a plain class — the template loads the
  adapter via `importlib.spec_from_file_location` without registering it in `sys.modules`, and Python
  3.12 `@dataclass` then errors in the spawned isolated worker;
  (2) `compare_outputs` made regime-aware (naive exact; radix order-tolerant sorted-set) — the baseline
  radix order is non-deterministic, so the template's per-workload exact compare wrongly marked radix
  rows INCORRECT. Authoritative correctness remains `bench/correctness.py` (valid-top-k, `matched_ratio==1.0`).

### Command + result
```
CUDA_VISIBLE_DEVICES=1 python3 bench/benchmark.py --device cuda:0 \
  --workloads bench/workloads.json --out bench/results.jsonl
-> 251 workloads: 250 PASSED, 1 INCORRECT (reg_ties_boundary, radix+exact-ties REGRESSION row;
   production=false; validated by correctness.py valid-top-k; not in the headline).
-> production headline (236 rows): geomean 0.9977, arithmetic 0.9979, min 0.80, max 1.097
   (candidate==baseline stub -> ~1.0 confirms a fair harness; ~±20% per-row tiny-kernel noise floor).
-> per-regime baseline median_us: decode n=212 p50 9.32 (7.63-13.59); radix n=24 p50 9.29 (8.22-74.79).
```
- **Immutable baseline frozen** in `bench/results.jsonl` (raw `samples_us` trimmed; stats retained).
  Provenance in `docs/benchmark_method.md`. No NCU this round (AC-11; the native candidate + a named
  bound come next). Build cached (no recompile).

