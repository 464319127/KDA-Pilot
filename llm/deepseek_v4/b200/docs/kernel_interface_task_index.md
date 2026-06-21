# deepseek-ai/DeepSeek-V4-Flash B200 Kernel Interface Task Index

- Generated at: `2026-06-21T18:14:29Z`
- Model slug: `deepseek_v4`
- Source capture dir: `/data/bbuf/kda-pilot/llm/deepseek_v4/b200/capture`
- Task count: `10`
- Evidence policy: runtime capture at SGLang kernel Python interfaces.

## Category Counts

| Category | Tasks |
|---|---:|
| `comm` | 4 |
| `norm` | 1 |
| `quant_gemm` | 1 |
| `quantization` | 4 |

## Tasks

| Task id | Category | Interface | Calls | Variants | Workloads |
|---|---|---|---:|---:|---|
| `jit_kernel_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2_custom_op` | `quantization` | `jit_kernel.per_token_group_quant_8bit_v2._per_token_group_quant_8bit_v2_custom_op` | 64728 | 698 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_per_token_group_quant_8bit_v2_per_token_group_quant_8bit_v2` | `quantization` | `jit_kernel.per_token_group_quant_8bit_v2.per_token_group_quant_8bit_v2` | 64728 | 698 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_quantization_fp8_kernel_deep_gemm_fp8_fp8_bf16_nt` | `quant_gemm` | `srt.layers.quantization.fp8_kernel.deep_gemm_fp8_fp8_bf16_nt` | 54752 | 698 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sglang_quant_method_fp8_linear_method_apply` | `quantization` | `sglang.quant_method.Fp8LinearMethod.apply` | 54752 | 837 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_all_reduce_get_custom_all_reduce_cls_custom_all_reduce_obj_real_all_reduce` | `comm` | `jit_kernel.all_reduce.get_custom_all_reduce_cls.CustomAllReduceObjReal.all_reduce` | 18096 | 116 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_outplace_all_reduce` | `comm` | `srt.distributed.parallel_state.outplace_all_reduce` | 18096 | 116 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_rmsnorm` | `norm` | `sgl_kernel.rmsnorm` | 10208 | 279 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sglang_quant_method_unquantized_linear_method_apply` | `quantization` | `sglang.quant_method.UnquantizedLinearMethod.apply` | 4872 | 139 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_inplace_all_reduce` | `comm` | `srt.distributed.parallel_state.inplace_all_reduce` | 2088 | 24 | `random_mid`, `random_high`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_reg_all_gather_into_tensor` | `comm` | `srt.distributed.parallel_state.reg_all_gather_into_tensor` | 232 | 120 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
