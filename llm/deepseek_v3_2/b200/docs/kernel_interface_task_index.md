# nvidia/DeepSeek-V3.2-NVFP4 B200 Kernel Interface Task Index

- Generated at: `2026-06-21T06:38:00Z`
- Model slug: `deepseek_v3_2`
- Source capture dir: `/data/bbuf/kda-pilot/llm/deepseek_v3_2/b200/capture`
- Task count: `20`
- Evidence policy: runtime capture at SGLang kernel Python interfaces.

## Category Counts

| Category | Tasks |
|---|---:|
| `attention` | 1 |
| `cache` | 1 |
| `comm` | 4 |
| `gemm` | 2 |
| `moe` | 1 |
| `norm` | 3 |
| `other` | 2 |
| `quant_gemm` | 1 |
| `quantization` | 3 |
| `rope` | 1 |
| `sampling` | 1 |

## Tasks

| Task id | Category | Interface | Calls | Variants | Workloads |
|---|---|---|---:|---:|---|
| `sglang_quant_method_unquantized_linear_method_apply` | `quantization` | `sglang.quant_method.UnquantizedLinearMethod.apply` | 50508 | 484 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_quantization_modelopt_quant_fp4_gemm` | `quant_gemm` | `srt.layers.quantization.modelopt_quant.fp4_gemm` | 28000 | 512 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sglang_quant_method_model_opt_fp4_linear_method_apply` | `quantization` | `sglang.quant_method.ModelOptFp4LinearMethod.apply` | 28000 | 512 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_quantization_fp4_utils_flashinfer_fp4_quantize_impl` | `quantization` | `srt.layers.quantization.fp4_utils._flashinfer_fp4_quantize_impl` | 28000 | 512 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_rmsnorm` | `norm` | `sgl_kernel.rmsnorm` | 27552 | 384 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_fused_add_rmsnorm` | `norm` | `sgl_kernel.fused_add_rmsnorm` | 27328 | 128 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_rope_apply_rope_inplace` | `rope` | `jit_kernel.rope.apply_rope_inplace` | 27328 | 256 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_hadamard_hadamard_transform` | `other` | `jit_kernel.hadamard.hadamard_transform` | 24888 | 216 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_all_reduce_get_custom_all_reduce_cls_custom_all_reduce_obj_real_all_reduce` | `comm` | `jit_kernel.all_reduce.get_custom_all_reduce_cls.CustomAllReduceObjReal.all_reduce` | 24600 | 104 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_outplace_all_reduce` | `comm` | `srt.distributed.parallel_state.outplace_all_reduce` | 24600 | 104 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_attention_base_attn_backend_attention_backend_forward` | `attention` | `srt.layers.attention.base_attn_backend.AttentionBackend.forward` | 13664 | 224 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_fused_store_index_cache_fused_store_index_k_cache` | `cache` | `jit_kernel.fused_store_index_cache.fused_store_index_k_cache` | 13664 | 128 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_layernorm_layernorm` | `norm` | `srt.layers.layernorm.layernorm` | 13664 | 128 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_moe_topk_fused_topk_deepseek` | `moe` | `srt.layers.moe.topk.fused_topk_deepseek` | 12992 | 128 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_fast_topk_transform_fused` | `sampling` | `sgl_kernel.fast_topk_transform_fused` | 11224 | 184 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_dsv3_fused_a_gemm` | `gemm` | `sgl_kernel.dsv3_fused_a_gemm` | 4148 | 28 | `random_low`, `random_mid`, `random_high`, `sharegpt_low` |
| `srt_models_deepseek_v2_flashinfer_dsv3_router_gemm` | `gemm` | `srt.models.deepseek_v2.flashinfer_dsv3_router_gemm` | 3944 | 28 | `random_low`, `random_mid`, `random_high`, `sharegpt_low` |
| `srt_distributed_parallel_state_inplace_all_reduce` | `comm` | `srt.distributed.parallel_state.inplace_all_reduce` | 2952 | 24 | `random_mid`, `random_high`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_activation_run_activation_inplace` | `other` | `jit_kernel.activation._run_activation_inplace` | 672 | 128 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_reg_all_gather_into_tensor` | `comm` | `srt.distributed.parallel_state.reg_all_gather_into_tensor` | 224 | 120 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
