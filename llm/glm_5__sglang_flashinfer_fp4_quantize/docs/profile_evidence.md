# Profile evidence - glm_5__sglang_flashinfer_fp4_quantize

**Standalone kernel target: 13.1% of total serving GPU time** (max across scenarios) on
`nvidia/GLM-5-NVFP4`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Kernel API shapes below are frozen from a one-time real `nvidia/GLM-5-NVFP4` production-path capture and replace the old noisy profiler shape strings.

- Model: `nvidia/GLM-5-NVFP4` (slug `glm_5`, tp=4)
- Python interface(s): `sglang.srt.layers.quantization.fp4_utils.fp4_quantize`
- Kernel family: `attention`  .  Category: `attention`
- GPU kernel(s): `fmhaSm100fKernel_QkvBfloat16OBfloat16H256SeparateQkvCausalVarSeqQ128Kv128PersistentContext`, `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1Mult`, `fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1VarSeqQ16Kv128P`, `kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPer`, `kernel_cutlass_kernel_flashinferquantizationkernelsnvfp4_quantizeNVFP4QuantizeSwizzledKern`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 11.86% |
| random | conc 32 | 2.03% |
| random | conc 100 | 4.89% |
| sharegpt | conc 1 | 11.40% |
| sharegpt | conc 32 | 8.39% |
| sharegpt | conc 100 | 13.10% |

**Peak: 13.1% in `sharegpt_high`.**

## Fresh captured kernel API shapes

- Shape source: `docs/captured_kernel_api_shapes.json`
- Standalone workloads: `bench/workloads.json`
- Workload count: 40
- Capture note: Captured 2026-07-07 on Verda B300 light-face-hides-fin-03-1 from a real nvidia/GLM-5-NVFP4 TP=4 SGLang production-path execution in temporary container sglang-glm5. Used a model-local HF cache after snapshot download, quantization=modelopt_fp4, kv_cache_dtype=fp8_e4m3, attention_backend=dsa with TRTLLM DSA prefill/decode as selected by SGLang, moe_runner_backend=flashinfer_trtllm, disabled FlashInfer autotune, disabled CUDA graph prefill/decode, and marked five request windows covering long prefill, short decode, mid concurrency, high concurrency, and an extra-long DSA/indexer prompt. The legacy fast_hadamard task maps to the captured DSA backend forward APIs because the current cookbook path did not emit a standalone hadamard_transform Python API; the old fast_hadamard GPU work is fused inside the DSA path.

Functions covered:
- `sglang.srt.layers.quantization.fp4_utils.fp4_quantize`

The old profiler `input_shapes` strings were noisy and are no longer an acceptance source.
Use the task-local workload file above for standalone single-GPU correctness and benchmark work.

## Validation Policy

Normal RLCR kernel work is a standalone single-GPU optimization task. Use the
captured workload set above for correctness and benchmark acceptance on one idle
target GPU, and do not add external runtime-readiness or fleet-level A/B gates to
the task loop.
