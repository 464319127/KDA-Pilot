# hunyuan3_preview — standalone kernel task selection

- Model: `tencent/Hy3-preview` (tp=8)
- Serving capture cmd (provenance only): `sglang serve --model-path tencent/Hy3-preview --tp 8 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1`
- Task mode: standalone single-GPU kernel optimization; no live serve, run_capture, or multi-GPU e2e gate during RLCR.
- Kept: max serving-profile GPU-time share `>= 3.0%`, non-comm, non-trtllm-MoE

| task | category | family | max % GPU | peak scenario | clean op |
|---|---|---|---:|---|---|
| `hunyuan3_preview__sglang_inplace_fused_experts` | moe | fused_moe_triton | 34.7% | random_mid | yes |
| `hunyuan3_preview__linear_gemm` | gemm | linear_gemm | 8.4% | random_mid | role |

## Dropped < 3.0%

- attention: 2.8%
- void_moe_sum_reduce_kernel_warp: 2.8%
- fused_add_rmsnorm: 2.2%

## Excluded (comm / trtllm fused-MoE)

- void sglang::cross_device_reduce_1stage<__half, 8>(sgla (memory_bound, comm): up to 59.5%
- void sglang::cross_device_reduce_2stage<__half, 8>(sgla (memory_bound, comm): up to 40.1%
- ncclDevKernel_AllReduce_Sum_f16_RING_LL(ncclDevKernelAr (comm, comm): up to 9.8%
- void cutlass::Kernel2<cutlass_80_simt_sgemm_64x64_8x5_t (gemm, comm): up to 8.3%
- cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_64x128x1 (gemm, comm): up to 2.0%
