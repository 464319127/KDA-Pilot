# mistralai/Ministral-3-14B-Instruct-2512 B200 Kernel Interface Task Index

- Generated at: `2026-06-21T09:00:41Z`
- Model slug: `ministral_3`
- Source capture dir: `/data/bbuf/kda-pilot-llm-interface/llm/ministral_3/b200/capture`
- Task count: `8`
- Evidence policy: runtime capture at SGLang kernel Python interfaces.

## Category Counts

| Category | Tasks |
|---|---:|
| `attention` | 1 |
| `cache` | 1 |
| `norm` | 2 |
| `other` | 1 |
| `quant_gemm` | 1 |
| `quantization` | 1 |
| `rope` | 1 |

## Tasks

| Task id | Category | Interface | Calls | Variants | Workloads |
|---|---|---|---:|---:|---|
| `sgl_kernel_fp8_scaled_mm` | `quant_gemm` | `sgl_kernel.fp8_scaled_mm` | 9148 | 124 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sglang_quant_method_fp8_linear_method_apply` | `quantization` | `sglang.quant_method.Fp8LinearMethod.apply` | 9148 | 124 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_fused_add_rmsnorm` | `norm` | `sgl_kernel.fused_add_rmsnorm` | 4575 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `srt_layers_attention_base_attn_backend_attention_backend_forward` | `attention` | `srt.layers.attention.base_attn_backend.AttentionBackend.forward` | 2287 | 58 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_kvcache_store_cache` | `cache` | `jit_kernel.kvcache.store_cache` | 2287 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_activation_run_activation_inplace` | `other` | `jit_kernel.activation._run_activation_inplace` | 2287 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `jit_kernel_rope_apply_rope_inplace` | `rope` | `jit_kernel.rope.apply_rope_inplace` | 2287 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
| `sgl_kernel_rmsnorm` | `norm` | `sgl_kernel.rmsnorm` | 57 | 31 | `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high` |
