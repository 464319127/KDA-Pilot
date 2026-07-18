# Profile evidence - mistral_small4__attention

**Standalone kernel target: 8.3% of total serving GPU time** (max across scenarios) on
`mistralai/Mistral-Small-4-119B-2603`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `mistralai/Mistral-Small-4-119B-2603` production-path capture and replace the old noisy profiler shape strings.

- Model: `mistralai/Mistral-Small-4-119B-2603` (slug `mistral_small4`, tp=1)
- Python interface(s): `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`, `flashinfer.prefill.trtllm_ragged_attention_deepseek`
- Kernel family: `attention`  .  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvBfloat16OBfloat16H128SeparateQkvCausalVarSeqQ128Kv128PersistentContext`, `fmhaSm100fKernel_QkvBfloat16OBfloat16H128SeparateQkvDenseVarSeqQ128Kv128PersistentContext`, `fmhaSm100fKernel_QkvBfloat16OBfloat16HQk320HV256PagedKvDenseP64MultiCtasKvCgaVarSeqQ16Kv12`, `fmhaSm100fKernel_QkvBfloat16OBfloat16HQk320HV256PagedKvDenseP64MultiCtasKvVarSeqQ16Kv128St`, `fmhaSm100fKernel_QkvBfloat16OBfloat16HQk320HV256PagedKvDenseP64VarSeqQ16Kv128PersistentSwa`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 6.20% |
| random | conc 32 | 2.68% |
| random | conc 100 | 6.13% |
| sharegpt | conc 1 | 6.44% |
| sharegpt | conc 32 | 8.32% |
| sharegpt | conc 100 | 7.03% |

**Peak: 8.3% in `sharegpt_mid`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 36
- Capture note: Captured 2026-07-08 on Verda B300 light-face-hides-fin-03-1 from a real mistralai/Mistral-Small-4-119B-2603 TP=1 SGLang production-path execution in temporary container sglang-mistral-small4 with TRTLLM/MLA attention and FP8 GEMM; CUDA graph disabled only to keep the temporary shape-capture hook out of graph capture; request windows include long/short prompt and mid/high concurrency serving paths.

Functions covered:
- `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`
- `flashinfer.prefill.trtllm_ragged_attention_deepseek`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
