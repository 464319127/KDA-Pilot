# nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 B200 Kernel Interface Task Index

- Generated at: `2026-06-22T01:19:30Z`
- Model slug: `nemotron3_ultra`
- Source capture dir: `/data/bbuf/kda-pilot/llm/nemotron3_ultra/b200/capture`
- Task count: `14`
- Evidence policy: runtime capture at SGLang kernel Python interfaces.

## Category Counts

| Category | Tasks |
|---|---:|
| `attention` | 2 |
| `cache` | 1 |
| `comm` | 4 |
| `moe` | 3 |
| `norm` | 2 |
| `other` | 1 |
| `quantization` | 1 |

## Tasks

| Task id | Category | Interface | Calls | Variants | Workloads |
|---|---|---|---:|---:|---|
| `sglang_quant_method_unquantized_linear_method_apply` | `quantization` | `sglang.quant_method.UnquantizedLinearMethod.apply` | 167224 | 2240 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_fused_add_rmsnorm` | `norm` | `sgl_kernel.fused_add_rmsnorm` | 57888 | 280 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_all_reduce_get_custom_all_reduce_cls_custom_all_reduce_obj_real_all_reduce` | `comm` | `jit_kernel.all_reduce.get_custom_all_reduce_cls.CustomAllReduceObjReal.all_reduce` | 53184 | 232 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_outplace_all_reduce` | `comm` | `srt.distributed.parallel_state.outplace_all_reduce` | 53184 | 232 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_moe_align_block_size` | `moe` | `sgl_kernel.moe_align_block_size` | 25728 | 280 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_moe_moe_runner_triton_utils_fused_moe_inplace_fused_experts` | `moe` | `srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts` | 25728 | 280 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_activation_run_unary_activation_inplace` | `other` | `jit_kernel.activation._run_unary_activation_inplace` | 25728 | 280 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_kvcache_store_cache` | `cache` | `jit_kernel.kvcache.store_cache` | 6432 | 280 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_moe_sum_reduce` | `moe` | `sgl_kernel.moe_sum_reduce` | 5376 | 112 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_inplace_all_reduce` | `comm` | `srt.distributed.parallel_state.inplace_all_reduce` | 5232 | 48 | `random_mid`, `random_high`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_attention_flashinfer_backend_flash_infer_attn_backend_forward_decode` | `attention` | `srt.layers.attention.flashinfer_backend.FlashInferAttnBackend.forward_decode` | 5088 | 424 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_attention_flashinfer_backend_flash_infer_attn_backend_forward_extend` | `attention` | `srt.layers.attention.flashinfer_backend.FlashInferAttnBackend.forward_extend` | 1344 | 112 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_reg_all_gather_into_tensor` | `comm` | `srt.distributed.parallel_state.reg_all_gather_into_tensor` | 536 | 240 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_rmsnorm` | `norm` | `sgl_kernel.rmsnorm` | 528 | 280 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
