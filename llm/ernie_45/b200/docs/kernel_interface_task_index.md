# baidu/ERNIE-4.5-21B-A3B-PT B200 Kernel Interface Task Index

- Generated at: `2026-06-21T08:20:25Z`
- Model slug: `ernie_45`
- Source capture dir: `/data/bbuf/kda-pilot-llm-interface/llm/ernie_45/b200/capture`
- Task count: `11`
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
| `sampling` | 1 |

## Tasks

| Task id | Category | Interface | Calls | Variants | Workloads |
|---|---|---|---:|---:|---|
| `sglang_quant_method_unquantized_linear_method_apply` | `quantization` | `sglang.quant_method.UnquantizedLinearMethod.apply` | 6344 | 186 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_fused_add_rmsnorm` | `norm` | `sgl_kernel.fused_add_rmsnorm` | 3173 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_activation_run_activation_inplace` | `other` | `jit_kernel.activation._run_activation_inplace` | 3116 | 93 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_attention_base_attn_backend_attention_backend_forward` | `attention` | `srt.layers.attention.base_attn_backend.AttentionBackend.forward` | 1586 | 57 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_kvcache_store_cache` | `cache` | `jit_kernel.kvcache.store_cache` | 1586 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_rope_apply_rope_inplace` | `rope` | `jit_kernel.rope.apply_rope_inplace` | 1586 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_moe_align_block_size` | `moe` | `sgl_kernel.moe_align_block_size` | 1530 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_moe_moe_runner_triton_utils_fused_moe_inplace_fused_experts` | `moe` | `srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts` | 1530 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_topk_softmax` | `sampling` | `sgl_kernel.topk_softmax` | 1530 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_moe_sum_reduce` | `moe` | `sgl_kernel.moe_sum_reduce` | 621 | 16 | `random_low`, `random_mid`, `random_high`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_rmsnorm` | `norm` | `sgl_kernel.rmsnorm` | 56 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
