# Run Log — `verify_tree_greedy` baseline-vs-candidate (B200)

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
| Python | 3.12.3 |
| Baseline upstream commit | `7e6587c94a1d0305815a14067c5d3cc02a9b0f36` (SGLang `main`) |
| Local commit at run time | `85795628d` |

### Source hashes (sha256)

| File | sha256 |
|------|--------|
| `solution/verify_tree_greedy_candidate.cuh` | `ec47c49e9fe36ff1db9fdfbc6a77d55bfb48394a90d4f9055056dd3b4a95538a` |
| `bench/verify_tree_greedy_ext.cu` | `f5baa1bdcac249d0d1608d6a4d324c0f1538ef9d2a2d15cebc0f75a8dafcbe30` |
| `baseline/verify_tree_greedy_kernel.cuh` | `d4bbf9d770e95b9e777742b82ed7b4a3651b0eb161d24dff534017512a603b2b` |

## GPU idle evidence (pinned GPU 7)

`nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv` (GPU 7 row):

- **Before**: `7, 0 %, 0 MiB` (idle — no active compute, no meaningful memory).
- **After**: `7, 0 %, 4 MiB` (idle; 4 MiB trivial residual context).

Other GPUs (2: 82 GiB, 5: 156 GiB) were in use by other tenants throughout; only GPU 7 was used for this run.

## Commands

Workspace synced into the container via `tar | docker exec` (see method notes). Then:

```bash
# Build + exact-match correctness gate (candidate == baseline == independent oracle)
CUDA_VISIBLE_DEVICES=7 TORCH_CUDA_ARCH_LIST=10.0 python bench/correctness.py

# Amplified fair benchmark (CUDA-event GPU time authoritative)
CUDA_VISIBLE_DEVICES=7 TORCH_CUDA_ARCH_LIST=10.0 python bench/benchmark.py --device cuda:0
```

Benchmark settings (from `bench/benchmark.py` provenance record): `warmup_runs=10`,
`num_trials=7`, `inner_iterations_min/max=1/4096`, `target_sample_us=1000`,
`isolated=True`. Inner-loop amplification ramped to **256** back-to-back launches per
CUDA-event pair (256 × ~4 µs ≈ 1024 µs sample).

## Outcomes

- **Correctness**: 17/17 checks PASS — `candidate == baseline == oracle`, exact
  integer/structural match, across the upstream fixture, all 10 production shapes (5 seeds
  each), and 6 regression rows (incl. the `bs>CAP, nd>2` baseline-fallback row).
- **Benchmark**: 16/16 workloads PASS; production equal-weight geomean speedup = **0.9956**
  (candidate ≈ baseline ≈ ~4.0–4.8 µs/launch). See `docs/results.md`.

Raw per-run records: `bench/results.jsonl` (kept local, NOT staged for the PR per the
PR-scope rules).
