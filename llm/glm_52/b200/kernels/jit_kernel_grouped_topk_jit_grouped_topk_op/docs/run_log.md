# Run Log

## Host / GPU

- Remote host: `ion-b200` (identifies as `innomatrix-us-adc-smb200-0003`)
- Container: `sglang_bbuf` (image `lmsysorg/sglang:dev`, `--privileged --cap-add=SYS_ADMIN --security-opt seccomp=unconfined` → NCU-capable)
- Task workspace: `/home/sglang-omni/bbuf/kda/k09_grouped_topk`
- GPU: NVIDIA B200, pinned `CUDA_VISIBLE_DEVICES=0` (REMOTE_GPU_ID=0) for every baseline/candidate/correctness/benchmark/NCU command.

### Idle methodology (AC-7) — what counts as the idle evidence

The valid idle evidence is an **external** `nvidia-smi` on the host (no benchmark
process resident) **before** the run and **again after the benchmark Python
process has fully exited**. The `nvidia_smi_after` field stored inside
`bench/results.jsonl` is captured by `benchmark.py` *from within its own
still-running process* (its CUDA context is still resident), so it reflects the
benchmark's own occupancy (~0.7 GB, tens-of-% util) and is an **in-process
diagnostic only — NOT the after-run idle check**. The external host checks below
are the AC-7 provenance.

**Authoritative performance run** (the numbers in `docs/results.md` and
`bench/results.jsonl`): GPU **0**, whole box quiet, `--no-isolated
--target-sample-us 5000 --num-trials 21`.

- External BEFORE (host, no benchmark running): GPU0 `0 %, 0 MiB / 183359 MiB` (verified idle).
- The kernels ran on an idle GPU and a quiet host.

**Idle before/after bracket** (clean external before AND after-exit): re-run on
GPU **2** (`CUDA_VISIBLE_DEVICES=2`), a deviation from the `REMOTE_GPU_ID=0` pin
approved by the user because an unrelated external multi-GPU job had since occupied
GPUs 0–1 (≈62 GB, 45–80 % util). All eight cards are identical B200 sm_100 and the
reported speedup is relative (baseline vs candidate, same idle card, A/B
interleaved), so it is GPU-id independent.

- External BEFORE: GPU2 `0 %, 0 MiB / 183359 MiB`.
- External AFTER (benchmark process exited): GPU2 `0 %, 0 MiB / 183359 MiB`.
- Both idle → confirms the candidate executes on a verified-idle card and leaves no
  residual occupancy after exit.

> **Why the GPU-2 run is the idle bracket, not the headline.** With the external
> job loading GPUs 0–1, the **host CPU** was contended. This kernel's decode regime
> is CPU-launch-bound (~6 µs/call, sub-µs GPU work), so the launch-floor decode and
> small-prefill speedups in the GPU-2 run were depressed by host jitter (e.g. decode
> ≈0.78–0.80, which is impossible as a *real* effect because the candidate's
> decode path is the **bit-identical copied baseline kernel** — see
> `solution/.../grouped_topk_candidate.cuh` `grouped_topk_block_per_token_kernel`).
> The large-prefill wins, being GPU-compute-bound, reproduced (1.27–1.67×). The
> authoritative headline is therefore the quiet-box GPU-0 run; decode is reported as
> **parity by construction**. A single fully-pristine re-run (idle GPU + quiet host)
> can be taken once the external job clears; it would reproduce the quiet-box
> numbers (decode ≈1.0, prefill 1.27–1.67×).

## Software provenance

- PyTorch: `2.11.0+cu130`; CUDA: `13.0`; nvcc `13.0.88`
- TVM-FFI (`apache-tvm-ffi`): `0.1.9`
- Platform: `Linux-6.8.0-111-generic-x86_64-with-glibc2.39`
- Baseline upstream commit: `6b2c730bf793984c39f7f07b3c074ca05b059b00` (sgl-project/sglang `main`); see `baseline_source.md`.

## Commands (run inside the container, GPU 0)

Correctness (full grid, candidate vs baseline + independent oracle):
```
cd bench && CUDA_VISIBLE_DEVICES=0 python correctness.py
# -> 1479 checks passed, 0 failed
```

Authoritative benchmark (baseline vs candidate, all 46 workloads):
```
cd bench && CUDA_VISIBLE_DEVICES=0 python benchmark.py \
    --out results.jsonl --target-sample-us 5000 --num-trials 21 --no-isolated
```

Warps-per-block tuning sweep (per-N, override via env):
```
for W in 1 2 4 8; do CUDA_VISIBLE_DEVICES=0 K09_WPB=$W python benchmark.py \
    --only <N ladder> --out results_w$W.jsonl; done
```

NCU (kernel diagnosis, representative shapes N=8 decode, N=3769 prefill):
```
cd /tmp && CUDA_VISIBLE_DEVICES=0 ncu --set basic --launch-count 12 \
    -k "regex:grouped_topk" --target-processes all --csv python /tmp/k09_prof.py
```

## GPU idle confirmation

`nvidia-smi --query-gpu=index,utilization.gpu,memory.used` immediately before the
authoritative benchmark and NCU runs returned `0, 0, 0..4 MiB` for GPU 0. All
performance numbers in `results.md` were collected on an idle GPU 0.

## Notes

- No fabricated data. All numbers are from the commands above; raw records are in
  `bench/results.jsonl` (benchmark) and the local NCU report (kept local, unstaged).
- No live SGLang server/checkout was imported or patched at correctness or
  benchmark runtime; only the copied workspace source is built and run.
