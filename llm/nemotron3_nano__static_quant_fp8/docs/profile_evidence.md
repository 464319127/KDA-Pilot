# Profile evidence — nemotron3_nano__static_quant_fp8

**Why this is a standalone kernel target:** it is **4.1% of total serving GPU time** (max across scenarios) on `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`, measured by profiling the exact
cookbook deployment. This is target-selection provenance and headroom context, not the validation path. Profiler role name; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` (slug `nemotron3_nano`, tp=1)
- Python interface: `<confirm via capture; profiler role=static_quant_fp8>`
- Category: `quant_gemm`
- GPU kernel(s): `_static_quant_fp8`
- Profiler op provenance: `aten::copy_`, `aten::empty`, `aten::to`

## % of GPU time by scenario

| dataset | scenario (concurrency) | % of GPU time |
|---|---|---|
| sharegpt_low | sharegpt (conc 1) | 4.13% |

**Peak: 4.1% in `sharegpt_low` (sharegpt, concurrency 1).**

## Input shapes (profiler)
- `[[1], [1], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
python3 -m sglang.launch_server --model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --trust-remote-code --max-running-requests 1024
```

Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
