# Profile evidence — nemotron3_nano__sglang_unified_attention_with_output

**Why this is a standalone kernel target:** it is **6.1% of total serving GPU time** (max across scenarios) on `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`, measured by profiling the exact
cookbook deployment. This is target-selection provenance and headroom context, not the validation path. Clean Python interface from profiler provenance.

- Model: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` (slug `nemotron3_nano`, tp=1)
- Python interface: `sglang.unified_attention_with_output`
- Category: `quant_gemm`
- GPU kernel(s): `nvjet_sm100_tst_128x256_64x6_2x1_2cta_v_bz_TNT`, `void flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::KernelTraits<(flashinfer::`
- Profiler op provenance: `aten::mm`, `sglang::unified_attention_with_output`

## % of GPU time by scenario

| dataset | scenario (concurrency) | % of GPU time |
|---|---|---|
| sharegpt_mid | sharegpt (conc 32) | 6.06% |

**Peak: 6.1% in `sharegpt_mid` (sharegpt, concurrency 32).**

## Input shapes (profiler)
- `[[13193, 2688], [2688, 10304]]`
- `[[13312, 4096], [13312, 2, 128], [13312, 2, 128], [13312, 4096], [], [], [], [],`

## Original serving capture command (provenance only)
```bash
python3 -m sglang.launch_server --model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 --trust-remote-code --max-running-requests 1024
```

Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
