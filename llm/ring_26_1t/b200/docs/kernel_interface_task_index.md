# inclusionAI/Ring-2.6-1T B200 Kernel Interface Task Index

- Generated at: `2026-06-21T14:01:04Z`
- Model slug: `ring_26_1t`
- Source capture dir: `/data/bbuf/kda-pilot/llm/ring_26_1t/b200/capture`
- Task count: `17`
- Evidence policy: runtime capture at SGLang kernel Python interfaces.

## Category Counts

| Category | Tasks |
|---|---:|
| `attention` | 1 |
| `comm` | 4 |
| `moe` | 4 |
| `norm` | 2 |
| `other` | 1 |
| `quant_gemm` | 2 |
| `quantization` | 2 |
| `rope` | 1 |

## Tasks

| Task id | Category | Interface | Calls | Variants | Workloads |
|---|---|---|---:|---:|---|
| `sgl_kernel_sgl_per_token_quant_fp8` | `quantization` | `sgl_kernel.sgl_per_token_quant_fp8` | 305310 | 1824 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_fp8_scaled_mm` | `quant_gemm` | `sgl_kernel.fp8_scaled_mm` | 221387 | 2720 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sglang_quant_method_compressed_tensors_linear_method_apply` | `quantization` | `sglang.quant_method.CompressedTensorsLinearMethod.apply` | 221387 | 2720 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_fused_add_rmsnorm` | `norm` | `sgl_kernel.fused_add_rmsnorm` | 88337 | 296 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_activation_run_activation_inplace` | `other` | `jit_kernel.activation._run_activation_inplace` | 86128 | 880 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_all_reduce_get_custom_all_reduce_cls_custom_all_reduce_obj_real_all_reduce` | `comm` | `jit_kernel.all_reduce.get_custom_all_reduce_cls.CustomAllReduceObjReal.all_reduce` | 78585 | 232 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_outplace_all_reduce` | `comm` | `srt.distributed.parallel_state.outplace_all_reduce` | 78585 | 232 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_rope_apply_rope_inplace` | `rope` | `jit_kernel.rope.apply_rope_inplace` | 44165 | 592 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_moe_align_block_size` | `moe` | `sgl_kernel.moe_align_block_size` | 41961 | 296 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_moe_moe_runner_triton_utils_fused_moe_inplace_fused_experts` | `moe` | `srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts` | 41960 | 296 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_moe_topk_fused_topk_deepseek` | `moe` | `srt.layers.moe.topk.fused_topk_deepseek` | 41960 | 296 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_moe_sum_reduce` | `moe` | `sgl_kernel.moe_sum_reduce` | 17024 | 168 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_rmsnorm` | `norm` | `sgl_kernel.rmsnorm` | 11597 | 888 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_inplace_all_reduce` | `comm` | `srt.distributed.parallel_state.inplace_all_reduce` | 10304 | 64 | `random_mid`, `random_high`, `sharegpt_mid`, `sharegpt_high` |
| `srt_models_deepseek_common_attention_forward_methods_forward_mla_bmm_fp8_op` | `attention` | `srt.models.deepseek_common.attention_forward_methods.forward_mla._bmm_fp8_op` | 9927 | 480 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_bmm_fp8` | `quant_gemm` | `sgl_kernel.bmm_fp8` | 9927 | 480 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_reg_all_gather_into_tensor` | `comm` | `srt.distributed.parallel_state.reg_all_gather_into_tensor` | 560 | 264 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
