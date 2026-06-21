# XiaomiMiMo/MiMo-V2.5 B200 Kernel Interface Task Index

- Generated at: `2026-06-21T11:16:34Z`
- Model slug: `mimo_v25`
- Source capture dir: `/data/bbuf/kda-pilot/llm/mimo_v25/b200/capture`
- Task count: `18`
- Evidence policy: runtime capture at SGLang kernel Python interfaces.

## Category Counts

| Category | Tasks |
|---|---:|
| `attention` | 1 |
| `comm` | 4 |
| `moe` | 3 |
| `norm` | 2 |
| `other` | 1 |
| `quant_gemm` | 1 |
| `quantization` | 4 |
| `rope` | 1 |
| `sampling` | 1 |

## Tasks

| Task id | Category | Interface | Calls | Variants | Workloads |
|---|---|---|---:|---:|---|
| `jit_kernel_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2_custom_op` | `quantization` | `jit_kernel.per_token_group_quant_8bit_v2._per_token_group_quant_8bit_v2_custom_op` | 33408 | 408 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2` | `quantization` | `jit_kernel.per_token_group_quant_8bit_v2.per_token_group_quant_8bit_v2` | 33408 | 408 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_fused_add_rmsnorm` | `norm` | `sgl_kernel.fused_add_rmsnorm` | 22272 | 136 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_all_reduce_get_custom_all_reduce_cls_custom_all_reduce_obj_real_all_reduce` | `comm` | `jit_kernel.all_reduce.get_custom_all_reduce_cls.CustomAllReduceObjReal.all_reduce` | 19788 | 108 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_outplace_all_reduce` | `comm` | `srt.distributed.parallel_state.outplace_all_reduce` | 19788 | 108 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_quantization_fp8_kernel_deep_gemm_fp8_fp8_bf16_nt` | `quant_gemm` | `srt.layers.quantization.fp8_kernel.deep_gemm_fp8_fp8_bf16_nt` | 11600 | 544 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sglang_quant_method_fp8_linear_method_apply` | `quantization` | `sglang.quant_method.Fp8LinearMethod.apply` | 11600 | 544 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_attention_base_attn_backend_attention_backend_forward` | `attention` | `srt.layers.attention.base_attn_backend.AttentionBackend.forward` | 11136 | 464 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_activation_run_activation_inplace` | `other` | `jit_kernel.activation._run_activation_inplace` | 11136 | 272 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sglang_quant_method_unquantized_linear_method_apply` | `quantization` | `sglang.quant_method.UnquantizedLinearMethod.apply` | 11136 | 136 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_rope_rotary_embedding_with_key` | `rope` | `jit_kernel.rope.rotary_embedding_with_key` | 11136 | 272 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_moe_align_block_size` | `moe` | `sgl_kernel.moe_align_block_size` | 10904 | 136 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_moe_moe_runner_triton_utils_fused_moe_inplace_fused_experts` | `moe` | `srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts` | 10904 | 136 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_grouped_topk_jit_grouped_topk_op` | `sampling` | `jit_kernel.grouped_topk._jit_grouped_topk_op` | 10904 | 136 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_moe_sum_reduce` | `moe` | `sgl_kernel.moe_sum_reduce` | 4324 | 64 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_inplace_all_reduce` | `comm` | `srt.distributed.parallel_state.inplace_all_reduce` | 2716 | 28 | `random_mid`, `random_high`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_reg_all_gather_into_tensor` | `comm` | `srt.distributed.parallel_state.reg_all_gather_into_tensor` | 232 | 124 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_rmsnorm` | `norm` | `sgl_kernel.rmsnorm` | 232 | 136 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
