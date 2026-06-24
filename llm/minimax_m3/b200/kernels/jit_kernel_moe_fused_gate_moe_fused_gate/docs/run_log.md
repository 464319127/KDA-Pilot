# Run Log — `moe_fused_gate` (B200)

## Environment
- Host: `ion-b200` (`innomatrix-us-adc-smb200-0003`), user `sglang-omni`.
- Container: `sglang_bbuf` (lmsysorg/sglang:dev), `--privileged --cap-add=SYS_ADMIN --security-opt seccomp=unconfined` (Nsight-capable).
- GPU: NVIDIA B200, **pinned `REMOTE_GPU_ID=4`** (`CUDA_VISIBLE_DEVICES=4`).
- Software: PyTorch 2.11.0+cu130, CUDA 13.0, nvcc 13.0, tvm_ffi (`/usr/local/lib/python3.12/dist-packages/tvm_ffi`), ncu at `/usr/local/cuda/bin/ncu`.
- Baseline upstream: SGLang `main` @ `34dd9c28caf4f7dd185e58e462a1344b52568e2e`.
- Task-owned remote workspace: `/home/sglang-omni/bbuf/kda_tasks/k05_moe_fused_gate` (synced from the local worktree; never writes into another task or a live SGLang checkout).

## GPU idle evidence
- Before measurement: GPU 4 = `0% util, 0 MiB` (GPUs 0/1 were busy with other users at 81%; GPUs 2-7 idle). GPU 4 confirmed idle before measuring.
- After measurement: GPU 4 = `34% util, 742 MiB` — this is THIS benchmark's own python/CUDA process; all other GPUs `0%`. No competing workload on GPU 4 during measurement.

## Build
- TVM-FFI `load_inline` (bench/_jit_build.py); exported `moe_fused_gate -> MoEFusedGateKernel::run`.
- Flags (symmetric, both sides): cflags `-std=c++20 -O3`; cuda `-DSGL_CUDA_ARCH=1000 -std=c++20 -O3 --expt-relaxed-constexpr`; `TVM_FFI_CUDA_ARCH_LIST=10.0` (B200 sm_100).
- Baseline compiles + loads cleanly.

## Baseline correctness (vs independent oracle)
- Prefill (large-token, M>512): matches oracle exactly (idx + weights) — e.g. shared slot idx=128, weight=1.0.
- Decode (small-token, M<=512): **UB bug** — uninitialized `warp_maxs[4..7]` read for num_experts=128 → `CUDA illegal memory access` on a cold context; nondeterministic. When warmed it matches the oracle (M=1/32/79/512). See docs/baseline_source.md. The baseline decode path is therefore not reliably runnable, and decode baseline timing is not obtainable.

## Immutable baseline numbers — PREFILL (CUDA-event median, num_trials=7, --no-isolated, target-sample-us=1000)
Command: `CUDA_VISIBLE_DEVICES=4 python benchmark.py --only <prefill ids> --num-trials 7 --no-isolated`
(candidate column == baseline here because the candidate was not yet built; this row set is the frozen baseline reference.)

| Workload (M, E=128, topk=5) | baseline median (us) |
|---|---|
| m1074 | 6.157 |
| m2340 | 6.160 |
| m4004 | 6.350 |
| m4339 | 6.424 |
| m4951 | 8.218 |
| m5398 | 8.222 |
| m5956 | 8.227 |
| m7120 | 8.233 |
| m7149 | 8.230 |
| m7299 | 8.263 |
| m7432 | 8.240 |

Observation: the kernel is tiny and launch/overhead-bound even at prefill (~6 us for M<=4339, ~8 us for M>=4951; the step near M~4500 reflects occupancy/grid changes). Decode (M<=79) is expected to be at the kernel-launch floor.

## Command log (round 0)
- Resolve commit: `git ls-remote https://github.com/sgl-project/sglang.git refs/heads/main` -> 34dd9c28.
- Recover source: sparse-checkout `python/sglang/jit_kernel/{include,csrc/moe}` @ 34dd9c28 -> baseline/.
- Sync workspace -> remote task dir; build baseline; run correctness.py (cold decode -> illegal access; warmed decode -> matches oracle).
- Capture baseline prefill numbers (above).
- Build candidate (warp-per-token, cold-safe); smoke test M=1 as cold first launch -> correct (no crash).
- `python bench/correctness.py` -> **490 checks pass, 0 fail** (candidate-vs-oracle all paths;
  candidate-vs-baseline prefill; off-domain E=256 fallback candidate==baseline==oracle; subnormal; ties; M=0; determinism).
- `python benchmark.py --only <11 prefill ids> --num-trials 9 --no-isolated` -> candidate-vs-baseline geomean 1.0006 (parity).
- `python bench/bench_decode_candidate.py` -> candidate decode ~4.10us (M=1..12), ~4.6us (M=79), 6.14us (M=512); baseline decode UB (not benchmarkable).
- `ncu --set basic` on candidate -> SM 0.06% / DRAM 0.02% / mem ~3% / occ 12.5% -> launch/latency-bound. Raw report at remote `/tmp/k05_cand_ncu.ncu-rep` (kept local, NOT staged).
- Reproduced baseline decode M=1 cold: ran OK on one probe, crashed (`illegal memory access`) on the correctness-grid cold first-launch -> nondeterministic UB.
- Independent Codex review (gpt-5.5:high) -> NEEDS-CHANGES (overclaimed fallback safety + evidence scope); applied all 5 must-fix items (dispatch safety scope, 490-check scope, decode command, off-domain + subnormal tests, NaN/Inf out-of-contract). Re-validated 490/490.
