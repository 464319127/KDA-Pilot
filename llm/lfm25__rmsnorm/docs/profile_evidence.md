# Profile evidence — lfm25__rmsnorm

**Why this is a standalone kernel target:** it is **5.2% of total serving GPU time** (max across scenarios) on `LiquidAI/LFM2.5-8B-A1B`, measured by profiling the exact
cookbook deployment. This is target-selection provenance and headroom context, not the validation path. Profiler role name; confirm exact Python interface via SGLANG_KERNEL_API_LOGLEVEL capture.

- Model: `LiquidAI/LFM2.5-8B-A1B` (slug `lfm25`, tp=1)
- Python interface: `<confirm via capture; profiler role=rmsnorm>`
- Category: `gemm`
- GPU kernel(s): `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object_at__tensorptrbf16gm`
- Profiler op provenance: `aten::_index_put_impl_`, `aten::_unsafe_view`, `aten::cat`, `aten::copy_`, `aten::empty`, `aten::empty_strided`

## % of GPU time by scenario

| dataset | scenario (concurrency) | % of GPU time |
|---|---|---|
| random_low | random (conc 1) | 5.17% |
| random_mid | random (conc 32) | 2.21% |
| sharegpt_low | sharegpt (conc 1) | 4.85% |
| sharegpt_mid | sharegpt (conc 32) | 2.11% |

**Peak: 5.2% in `random_low` (random, concurrency 1).**

## Input shapes (profiler)
- `[[18, 466682, 2048, 2], [], [18, 1, 2048, 2], [], []]`
- `[[1], [1], []]`
- `[[57544, 64], []]`
- `[[7193, 6144], [], []]`
- `[[7505, 2048], [2048, 32]]`
- `[[7505, 8, 64], []]`
- `[[[6222173], [101]], []]`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
sglang serve --model-path LiquidAI/LFM2.5-8B-A1B --tp 1 --attention-backend flashinfer --reasoning-parser qwen3 --tool-call-parser lfm2
```

Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
