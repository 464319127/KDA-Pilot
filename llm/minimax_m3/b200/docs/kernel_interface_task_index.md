# MiniMaxAI/MiniMax-M3-MXFP8 B200 Kernel Interface Task Index

- Generated at: `2026-06-21T16:35:46Z`
- Model slug: `minimax_m3`
- Source capture dir: `/data/bbuf/kda-pilot/llm/minimax_m3/b200/capture`
- Task count: `15`
- Evidence policy: runtime capture at SGLang kernel Python interfaces.

## Category Counts

| Category | Tasks |
|---|---:|
| `attention` | 3 |
| `cache` | 1 |
| `comm` | 4 |
| `moe` | 1 |
| `norm` | 3 |
| `quant_gemm` | 1 |
| `quantization` | 2 |

## Tasks

| Task id | Category | Interface | Calls | Variants | Workloads |
|---|---|---|---:|---:|---|
| `srt_layers_quantization_fp8_utils_triton_mxfp8_block_scaled_matmul` | `quant_gemm` | `srt.layers.quantization.fp8_utils.triton_mxfp8_block_scaled_matmul` | 69985 | 640 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sglang_quant_method_fp8_linear_method_apply` | `quantization` | `sglang.quant_method.Fp8LinearMethod.apply` | 69985 | 1480 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_quantization_fp8_utils_triton_mxfp8_blockscaled_linear` | `quantization` | `srt.layers.quantization.fp8_utils.triton_mxfp8_blockscaled_linear` | 69985 | 1480 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_gemma_fused_add_rmsnorm` | `norm` | `sgl_kernel.gemma_fused_add_rmsnorm` | 66681 | 296 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_all_reduce_get_custom_all_reduce_cls_custom_all_reduce_obj_real_all_reduce` | `comm` | `jit_kernel.all_reduce.get_custom_all_reduce_cls.CustomAllReduceObjReal.all_reduce` | 57553 | 216 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_outplace_all_reduce` | `comm` | `srt.distributed.parallel_state.outplace_all_reduce` | 57553 | 216 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_attention_base_attn_backend_attention_backend_forward` | `attention` | `srt.layers.attention.base_attn_backend.AttentionBackend.forward` | 33336 | 1112 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_minimax_qknorm_rope_fused_gemma_qknorm_rope` | `norm` | `jit_kernel.minimax_qknorm_rope._fused_gemma_qknorm_rope` | 33336 | 592 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_moe_fused_gate_moe_fused_gate` | `moe` | `jit_kernel.moe_fused_gate.moe_fused_gate` | 31686 | 296 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_inplace_all_reduce` | `comm` | `srt.distributed.parallel_state.inplace_all_reduce` | 9680 | 80 | `random_mid`, `random_high`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_flash_attention_v4_flash_attn_varlen_func` | `attention` | `jit_kernel.flash_attention_v4.flash_attn_varlen_func` | 1656 | 296 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_flash_attention_v4_flash_attn_with_kvcache` | `attention` | `jit_kernel.flash_attention_v4.flash_attn_with_kvcache` | 1656 | 296 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_kvcache_store_cache` | `cache` | `jit_kernel.kvcache.store_cache` | 1656 | 296 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_distributed_parallel_state_reg_all_gather_into_tensor` | `comm` | `srt.distributed.parallel_state.reg_all_gather_into_tensor` | 560 | 240 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_gemma_rmsnorm` | `norm` | `sgl_kernel.gemma_rmsnorm` | 552 | 296 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
