# Run Log — `sgl_kernel.topk_sigmoid` on B200

## Environment

| Field | Value |
|-------|-------|
| Host | `ion-b200` (`innomatrix-us-adc-smb200-0003`), user `sglang-omni` |
| Container | `sglang_bbuf` (image `lmsysorg/sglang:dev`, `--privileged --cap-add=SYS_ADMIN`, ncu-capable) |
| Remote workspace | `/home/sglang-omni/bbuf/kda_tasks/topk_sigmoid` (task-owned; streamed from the local kernel folder) |
| GPU | NVIDIA B200, sm_100 (CC 10.0), 183 GB |
| PyTorch | 2.11.0+cu130 | 
| CUDA / nvcc | 13.0 / V13.0.88 |
| tvm_ffi | 0.1.9 | 
| Python | 3.12 |
| Baseline commit | upstream SGLang `main` @ `5e6d7c1615a95dc5f98e69b4b18af0ae160b10b8` |

## Source hashes (candidate + ABI)

```
08d131b87b55f9b1740c1a78020b014a928ca812fc855cc31a37a56f21125c91  solution/topk_sigmoid_candidate.cuh
c9cae22be6109a365c8cc386e2db0922b48d8973f3f989262878c7b012b12a74  solution/topk_sigmoid_ext.cu
a32e5c66ed7d8a217b7a5d02455b49b92f85caa46eae7fcb91ddd17f9b034032  bench/csrc/topk_sigmoid_ext.h
4b8eadf84561c8b71c31893508caacc2c132a15510c725385616775a59e16ce3  baseline/topk_sigmoid_baseline.cu  (vendored verbatim)
```

## GPU selection and idle evidence

This task is pinned to **GPU 1** (`REMOTE_GPU_ID=1`). GPU 1 was **busy throughout this run** —
observed at 40% util / 62 GB, then 9% / ~100 GB resident, then 82% / 65 GB (a neighbor job on this
shared box). Per the task's GPU-idle policy, **timed** measurement requires an idle card, so the
user was asked before measuring elsewhere and **approved using an idle GPU**. The benchmark, floor
probe, and NCU therefore ran on **GPU 2** (idle), with both baseline and candidate on the same GPU
(fair A/B). Build + correctness are functional checks (contention-tolerant) and ran on GPU 1.

GPU 2 idle evidence (`nvidia-smi`, index, util%, mem MiB):

```
Before benchmark:  2, 0 %, 0 MiB
After  benchmark:  2, 0 %, 4 MiB
During run GPUs 0 and 1 were 52–82% util / 65 GB (other jobs); GPUs 3–7 idle.
```

## Key commands (run inside the container; `<WS>` = remote workspace)

```bash
# build (functional, GPU 1)
cd <WS>/bench && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -c "import build_ext; build_ext.get_ext()"

# correctness (functional, GPU 1) -> 34/34 PASS
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python correctness.py

# benchmark (idle GPU 2) -> production geomean 1.4381x
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python benchmark.py --workloads workloads.json --out results.jsonl --num-trials 7 --warmup-runs 10

# launch-floor probe (idle GPU 2)
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python floor_probe.py

# NCU (idle GPU 2) @ N=16474, candidate + baseline kernels
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. ncu --set basic --launch-skip 6 --launch-count 3 \
  -o /tmp/topk_ncu --force-overwrite python profile_ncu.py
```

## Notes

- The local toolchain (macOS) has no CUDA; all build/correctness/benchmark/profile work is remote.
- Raw artifacts (`bench/.build/`, `bench/results.jsonl`, `/tmp/topk_ncu.ncu-rep`, logs) are kept
  remote/local for evidence and are **excluded from the PR**; only summarized numbers land in
  `docs/results.md`.
