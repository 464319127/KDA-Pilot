# Run Log — `verify_tree_greedy` baseline-vs-candidate (B200)

Round 1 run with the literal **TVM-FFI direct-symbol ABI** (supersedes the Round-0 pybind run).

## Environment

| Field | Value |
|-------|-------|
| Remote host | `ion-b200` (identifies as `innomatrix-us-adc-smb200-0003`), user `sglang-omni` |
| Container | `sglang_bbuf` (image `lmsysorg/sglang:dev`) |
| Task workspace (remote, task-owned) | `/home/sglang-omni/bbuf/kda/sgl_kernel_verify_tree_greedy` |
| GPU (pinned) | id **7**, NVIDIA B200 (183359 MiB) — selected via `CUDA_VISIBLE_DEVICES=7` (maps to `cuda:0`) |
| PyTorch | 2.11.0+cu130 |
| CUDA runtime | 13.0 |
| nvcc | release 13.0, V13.0.88 |
| tvm-ffi | 0.1.9 (headers `tvm_ffi/include`, `libtvm_ffi.so`); build via `tvm_ffi.cpp.load` |
| Python | 3.12.3 |
| Baseline upstream commit | `7e6587c94a1d0305815a14067c5d3cc02a9b0f36` (SGLang `main`) |
| Code provenance | Round 1 working tree; the source hashes below pin the exact built sources |

### Source hashes (sha256)

| File | sha256 |
|------|--------|
| `bench/verify_tree_greedy_ffi.cu` (TVM-FFI ABI, finalized) | `f483a3dee61cdb2de340c0e8111b1ce5e1851225e3f00fdc5fa77f09cca55c89` |
| `solution/verify_tree_greedy_candidate.cuh` | `18cfffabea1d5e6bb162b9cba04e98b5149fc9e0406fb16402883bd353b0eaeb` |
| `baseline/verify_tree_greedy_kernel.cuh` | `d4bbf9d770e95b9e777742b82ed7b4a3651b0eb161d24dff534017512a603b2b` |

## GPU idle evidence (pinned GPU 7)

`nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv` (GPU 7 row):

- **Before**: `7, 0 %, 0 MiB` (idle — no active compute, no meaningful memory).
- **After**: `7, 0 %, 4 MiB` (idle; 4 MiB trivial residual context).

The selected card (GPU 7) was idle before and after, as required. During the runs, other
tenants' GPUs were intermittently busy (this finalized run: GPU 2 ~47%; the prior R1 run:
GPU 0 ~88%, GPU 1 ~70%; some GPUs held memory). This node-level host contention raised the
absolute per-launch floor to ~4.9–5.9 µs (vs ~4.0–4.8 µs in the Round-0 run on a quieter
node). The benchmark samples baseline and candidate
**interleaved within each trial**, so both sides experience identical conditions and the
speedup *ratio* is unaffected; the floor's sensitivity to host contention (with GPU 7 work
unchanged) further reinforces that the kernel is launch/scheduler-bound, not body-bound.

## Commands

Workspace synced into the container via `tar | docker exec`. Then on GPU 7:

```bash
# Build (TVM-FFI via tvm_ffi.cpp.load) + exact-match gate (candidate == baseline == oracle)
CUDA_VISIBLE_DEVICES=7 TORCH_CUDA_ARCH_LIST=10.0 python bench/correctness.py

# Amplified fair benchmark (CUDA-event GPU time authoritative; wall-clock secondary)
CUDA_VISIBLE_DEVICES=7 TORCH_CUDA_ARCH_LIST=10.0 python bench/benchmark.py --device cuda:0
```

Benchmark settings (from the `bench/results.jsonl` provenance record): `warmup_runs=10`,
`num_trials=7`, `inner_iterations_min/max=1/4096`, `target_sample_us=1000`, `isolated=True`.
Inner-loop amplification ramped to **256** back-to-back launches per CUDA-event pair
(256 × ~5 µs ≈ 1280 µs sample).

## Outcomes

- **Correctness**: 17/17 checks PASS — `candidate == baseline == oracle`, exact
  integer/structural match, across the upstream fixture, all 10 production shapes (5 seeds
  each), and 6 regression rows (incl. the `nd>2` baseline-fallback row).
- **Benchmark**: 16/16 workloads PASS; production equal-weight geomean speedup = **0.9933**
  (candidate ≈ baseline ≈ ~4.9–5.9 µs/launch). See `docs/results.md` and `docs/dispatch.md`.

Raw per-run records: `bench/results.jsonl` (kept local, NOT staged for the PR per the
PR-scope rules).
