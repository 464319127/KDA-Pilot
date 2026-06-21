# google/gemma-4-26B-A4B-it B200 Kernel Interface Task Index

- Generated at: `2026-06-21T07:14:34Z`
- Model slug: `gemma4`
- Source capture dir: `/data/bbuf/kda-pilot/llm/gemma4/b200/capture`
- Task count: `10`
- Evidence policy: runtime capture at SGLang kernel Python interfaces.

## Category Counts

| Category | Tasks |
|---|---:|
| `attention` | 1 |
| `cache` | 1 |
| `moe` | 3 |
| `norm` | 2 |
| `other` | 1 |
| `quantization` | 1 |
| `rope` | 1 |

## Tasks

| Task id | Category | Interface | Calls | Variants | Workloads |
|---|---|---|---:|---:|---|
| `sglang_quant_method_unquantized_linear_method_apply` | `quantization` | `sglang.quant_method.UnquantizedLinearMethod.apply` | 8250 | 217 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_rmsnorm` | `norm` | `sgl_kernel.rmsnorm` | 6655 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_activation_run_activation_inplace` | `other` | `jit_kernel.activation._run_activation_inplace` | 3300 | 62 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_attention_base_attn_backend_attention_backend_forward` | `attention` | `srt.layers.attention.base_attn_backend.AttentionBackend.forward` | 1650 | 110 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_kvcache_store_cache` | `cache` | `jit_kernel.kvcache.store_cache` | 1650 | 62 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_moe_align_block_size` | `moe` | `sgl_kernel.moe_align_block_size` | 1650 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_moe_moe_runner_triton_utils_fused_moe_inplace_fused_experts` | `moe` | `srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts` | 1650 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_fused_add_rmsnorm` | `norm` | `sgl_kernel.fused_add_rmsnorm` | 1650 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_rope_apply_rope_inplace` | `rope` | `jit_kernel.rope.apply_rope_inplace` | 1650 | 62 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_moe_sum_reduce` | `moe` | `sgl_kernel.moe_sum_reduce` | 600 | 13 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
