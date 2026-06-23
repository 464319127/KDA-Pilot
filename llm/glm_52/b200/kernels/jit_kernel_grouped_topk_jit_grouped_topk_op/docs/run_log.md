# Run Log

## Host / GPU

- Remote host: `ion-b200` (identifies as `innomatrix-us-adc-smb200-0003`)
- Container: `sglang_bbuf` (image `lmsysorg/sglang:dev`, `--privileged --cap-add=SYS_ADMIN --security-opt seccomp=unconfined` → NCU-capable)
- Task workspace: `/home/sglang-omni/bbuf/kda/k09_grouped_topk`
- GPU: NVIDIA B200. The task pins `REMOTE_GPU_ID=0`; an external job occupied GPU 0
  during the benchmark window, so the **authoritative benchmark ran on idle GPU 6**
  under a user-approved plan revision (recorded in `goal-tracker.md`). Correctness
  was run on an idle GPU (GPU-independent); NCU diagnosis was on GPU 0 (earlier,
  while usable). All cards are identical B200 sm_100; the reported speedup is
  relative (baseline vs candidate, same idle card, A/B interleaved) and GPU-id
  independent.

### Idle methodology (AC-7) — what counts as the idle evidence

The valid idle evidence is an **external** `nvidia-smi` on the host (no benchmark
process resident) **before** the run and **again after the benchmark Python
process has fully exited**. The `nvidia_smi_after` field stored inside
`bench/results.jsonl` is captured by `benchmark.py` *from within its own
still-running process* (CUDA context resident), so it reflects the benchmark's own
occupancy (~0.7 GB) and is an **in-process diagnostic only — NOT the after-run idle
check**. The external host checks below are the AC-7 provenance.

**Authoritative benchmark** (the numbers in `docs/results.md` and
`bench/results.jsonl`): idle **GPU 6** (`CUDA_VISIBLE_DEVICES=6`), `--no-isolated
--target-sample-us 5000 --num-trials 21`.

- External BEFORE (host, no benchmark running): GPU6 `0 %, 0 MiB / 183359 MiB`.
- External AFTER (benchmark process exited): GPU6 `0 %, 0 MiB / 183359 MiB`.
- Both idle → the candidate executes on a verified-idle card and leaves no residual
  occupancy after exit. This is the AC-7 before/after idle bracket.

**Quiet-host reference run** (contention-free headline): GPU **0**, round 0, whole
box idle, external BEFORE GPU0 `0 %, 0 MiB` verified. Decode measured 0.999,
headline 1.217× / 1.037×. (An external after-exit snapshot for that run could not be
captured because the external job occupied the box immediately afterward — which is
exactly why the authoritative bracket was re-taken on idle GPU 6.)

> **Host-contention caveat.** During the GPU-6 authoritative run the external job
> still loaded GPU 0 and thus the **host CPU**. This kernel's decode/small-N regime
> is CPU-launch-bound (~6 µs/call, sub-µs GPU work), so those launch-floor rows were
> depressed (decode ≈0.76–0.79; small prefill N≤645 ≈0.83–0.90). That is a
> measurement artifact, not a kernel effect: the candidate's decode/small-N dispatch
> path is the **bit-identical copied baseline kernel** (`grouped_topk_block_per_token_kernel`),
> so candidate==baseline by construction (1693 correctness checks) and the true ratio
> is 1.0, as the quiet-host run measured (0.999). The GPU-compute-bound large-prefill
> win (N≥1464: 1.33–1.67×) reproduced on both runs. See BitLesson
> `BL-20260623-host-contention-launch-floor`.

## Software provenance

- PyTorch: `2.11.0+cu130`; CUDA: `13.0`; nvcc `13.0.88`
- TVM-FFI (`apache-tvm-ffi`): `0.1.9`
- Platform: `Linux-6.8.0-111-generic-x86_64-with-glibc2.39`
- Baseline upstream commit: `6b2c730bf793984c39f7f07b3c074ca05b059b00` (sgl-project/sglang `main`); see `baseline_source.md`.

## Commands (run inside the container, GPU 0)

Correctness (full grid, candidate vs baseline + independent oracle):
```
cd bench && CUDA_VISIBLE_DEVICES=0 python correctness.py
# -> 1693 checks passed, 0 failed
```
The grid covers: all 43 captured production shapes (×2 seeds) exact ordered-index +
weights (fp32 atol=rtol=1e-5) vs baseline and the independent oracle; **output
stride/contiguity match**; constructed value edges (exact ties → smaller index,
equal sigmoid+bias via different bias, negative bias, saturating/Inf logits, N=0,
max N=3769, renormalize=False); **off-domain fallback at N≥768** (E=128, E=512;
topk=1/4/7; renormalize=False; scaling_factor=0.5) exact-matching the baseline; and
**non-contiguous / non-fp32 rejection** (identical on both sides). It also spawns a
fresh process with **K09_WPB=4** and uses the profiler to assert the override does
not route off-domain inputs to the warp kernel (production-domain gate holds).
(Correctness is GPU-independent; this round it was run on an idle GPU while GPU 0
was externally occupied — the count/result is identical on any B200.)

Authoritative benchmark (baseline vs candidate, all 46 workloads) — on idle GPU 6
(plan revision; GPU 0 externally occupied):
```
# external nvidia-smi BEFORE (GPU6 idle), then:
cd bench && CUDA_VISIBLE_DEVICES=6 python benchmark.py \
    --out results.jsonl --target-sample-us 5000 --num-trials 21 --no-isolated
# external nvidia-smi AFTER process exit (GPU6 idle)
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

External `nvidia-smi --query-gpu=index,utilization.gpu,memory.used` immediately
before the authoritative benchmark AND immediately after the Python process exited
returned `6, 0, 0 MiB` (GPU 6 idle, both checks) — the AC-7 before/after bracket.
The quiet-host reference run (round 0) had external BEFORE `0, 0, 0 MiB` on GPU 0.
Performance numbers in `results.md` are from the idle-GPU-6 authoritative run; the
contention-free headline is corroborated by the quiet-host GPU-0 reference.

## Notes

- No fabricated data. All numbers are from the commands above; raw records are in
  `bench/results.jsonl` (benchmark) and the local NCU report (kept local, unstaged).
- No live SGLang server/checkout was imported or patched at correctness or
  benchmark runtime; only the copied workspace source is built and run.
