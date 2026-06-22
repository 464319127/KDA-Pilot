# Run Log

## Host / GPU

- Remote host: `ion-b200` (identifies as `innomatrix-us-adc-smb200-0003`)
- Container: `sglang_bbuf` (image `lmsysorg/sglang:dev`, `--privileged --cap-add=SYS_ADMIN --security-opt seccomp=unconfined` → NCU-capable)
- Task workspace: `/home/sglang-omni/bbuf/kda/k09_grouped_topk`
- GPU: NVIDIA B200, pinned `CUDA_VISIBLE_DEVICES=0` (REMOTE_GPU_ID=0) for every baseline/candidate/correctness/benchmark/NCU command.
- GPU-0 idle state (verified before each timing run): utilization 0%, memory 0–4 MiB used (of 183359 MiB). Verified again after runs — still idle. No other compute processes on GPU 0 (resident contexts were on GPUs 3/4/5 only).

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
