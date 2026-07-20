# 扩散模型正确性约定

[English](diffusion_correctness_contract.md) | 简体中文

本文档列出每个扩散模型任务除了
`diffusion_benchmark_shape_coverage.md` 中的生产工作负载之外还必须覆盖的标准回归网格。

使用这些网格填充任务本地的 `bench/correctness.py` 测试。测试可以基于复制到本地的
基线和独立的 PyTorch/数学 oracle 实现；任务基准测试运行时不得导入、修改或
monkey-patch SGLang。

## QKNorm + RoPE

任务：

- `b200_diffusion_qknorm_rope__multi_shape`
- `h200_diffusion_qknorm_rope__multi_shape`
- `b200_ltx2_qknorm_split_rope__bitwise`

标准来源：`python/sglang/jit_kernel/tests/diffusion/test_qknorm_rope.py`。

回归网格：

- `batch_size`：`[1, 4096]` 范围内的所有 2 的幂、`x + 1`，以及
  `[1, 9, 129, 257, 2049, 4097]`。
- `num_heads`：`[8, 16, 24, 32]`。
- `head_dim`：`[64, 128, 256]`。
- `rope_dim`：`{64: [64], 128: [64, 128], 256: [64, 128, 256]}`。
- `is_neox`：`[False, True]`；为 `True` 时，只要求有效的 rotary-lane 配置。
- `position_dtype`：`torch.int32` 和 `torch.int64`。
- `dtype`：`torch.bfloat16`。
- `eps`：`1e-6`。
- 容差：`atol=8e-2`、`rtol=1e-2`。

Oracle：SGLang 风格的 Q/K 归一化，然后执行使用 cos/sin cache 的 FlashInfer 风格
RoPE。

## Norm Infer

任务：

- `b200_diffusion_norm_infer__multi_shape`
- `h200_diffusion_norm_infer__multi_shape`

标准来源：
`python/sglang/jit_kernel/tests/diffusion/test_qwen_image_modulation.py`。

`norm.norm_infer` 的回归网格：

- `batch_size`：`[1, 2, 4]`。
- `seq_len`：`[6, 33, 128, 257]`。
- `hidden_size`：`[512, 1024, 1536, 3072]`。
- `dtype`：`torch.float16`、`torch.bfloat16`、`torch.float32`。
- `eps`：`1e-6`。
- `is_rms_norm`：显式参数；如果本地任务适配器同时支持 LayerNorm 和 RMSNorm，
  则两类行都要包含。
- 容差：非 fp32 使用 `atol=5e-2`、`rtol=5e-2`；fp32 使用 `1e-5`。

在由同一网格推导出的行数，以及 `(4096, 128)`、`(16384, 128)` 等生产 per-head
行上，交叉检查 `rmsnorm_onepass.triton_one_pass_rms_norm`。

## GroupNorm + SiLU

任务：

- `b200_diffusion_group_norm_silu__multi_shape`
- `h200_diffusion_group_norm_silu__multi_shape`

标准来源：
`python/sglang/jit_kernel/tests/diffusion/test_group_norm_silu.py`。

回归网格：

- 2D 图像用例：`(2, 64, 32, 32)`，`num_groups=32`。
- 3D 视频用例：`(1, 64, 4, 16, 16)`，`num_groups=32`。
- Token 用例：`(4, 128)`，`num_groups=32`。
- 大 tile 用例：`(1, 128, 20, 256, 256)`，`num_groups=32`。
- `dtype`：`torch.float16`、`torch.bfloat16`、`torch.float32`。
- 容差：
  - fp16：`atol=3e-3`、`rtol=3e-3`；
  - bf16：`atol=7e-2`、`rtol=2e-2`；
  - fp32：`atol=1e-5`、`rtol=1e-5`。

Oracle：`silu(group_norm(x, num_groups, weight, bias, eps=1e-5))`。

封装风格的 `apply_group_norm_silu` 路径应覆盖 fp16 和 bf16 的 2D、3D 行。

## Rotary Embedding

任务：

- `b200_diffusion_rotary_embedding__multi_shape`
- `h200_diffusion_rotary_embedding__multi_shape`

标准来源：`python/sglang/jit_kernel/tests/test_rope.py`。

标准 `apply_rotary_embedding` 的回归网格：

- `batch_size` / 总 token 数：`[1, 2048]` 范围内的 2 的幂，以及
  `[1, 129, 2048, 2049]`。
- `num_kv_heads`：`[1, 2, 8]`。
- `gqa_ratio`：`[1, 4, 8]`；`num_qo_heads = num_kv_heads * gqa_ratio`。
- `rope_dim`：`[64, 128, 256, 512]`。
- `is_neox`：`[False, True]`。
- `dtype`：`torch.bfloat16`。
- 当 position 属于本地适配器路径时，同时包含 `torch.int32` 和 `torch.int64`。
- 容差：`atol=1e-2`、`rtol=1e-2`。

Oracle：FlashInfer 风格的 `apply_rope_with_cos_sin_cache_inplace`。

对于 `apply_ltx2_split_rotary_emb`，使用
`diffusion_benchmark_shape_coverage.md` 中的生产 split-rotary 行作为回归约定。

对于 `b200_ltx2_qknorm_split_rope__bitwise`，不允许使用容差。候选必须与任务本地
PyTorch eager 基线位级相等：先执行 `q_norm(q)`、`k_norm(k)`，然后分别对 Q 和 K
执行 SGLang 风格的 `apply_split_rotary_emb`。使用 `torch.equal` 比较两个输出。

## Scale Shift

任务：

- `b200_diffusion_fuse_scale_shift__multi_shape`
- `h200_diffusion_fuse_scale_shift__multi_shape`
- `b200_ltx2_dual_modulate__bitwise`

标准来源：
`python/sglang/jit_kernel/tests/diffusion/test_qwen_image_modulation.py`。

回归网格：

- `batch_size`：`[1, 2, 4]`。
- `seq_len`：`[6, 33, 128, 257]`。
- `hidden_size`：`[512, 1024, 1536, 3072]`。
- `dtype`：`torch.float16`、`torch.bfloat16`、`torch.float32`。
- `eps`：`1e-6`。
- `scale0`、`shift0`、`gate0`、`scale1`、`shift1`、`gate1`：shape 均为
  `(B, C)`。
- `index`：shape 为 `(B, L)`，覆盖双调制选择路径。
- residual 和 residual-gate 行覆盖
  `fuse_residual_layernorm_scale_shift_gate_select01_kernel`。
- 容差：非 fp32 使用 `atol=5e-2`、`rtol=5e-2`；fp32 使用 `1e-5`。

简单的 `fuse_scale_shift_kernel` 还必须同时覆盖 2D `(B, C)` 和 4D
`(B, F, 1, C)` scale/shift 布局。

对于 `b200_ltx2_dual_modulate__bitwise`，不允许使用容差。对于显式双调制，以及从
`temb_scale_shift` 得到的 cross-attention 双调制，候选都必须与任务本地 PyTorch
eager 基线位级相等。使用 `torch.equal` 比较两个输出 Tensor。

## CuTe DSL Norm + Tanh + Mul + Add

任务：

- `b200_diffusion_cutedsl_norm_tanh_mul_add__multi_shape`
- `h200_diffusion_cutedsl_norm_tanh_mul_add__multi_shape`

标准来源：
`python/sglang/jit_kernel/tests/diffusion/test_fused_norm_scale_shift.py`。

回归网格：

- `SHAPES = [(B, S, F, D)]`：`[(1, 1024, 8, 3072), (4, 512, 16, 3072)]`。
- `dtype`：`torch.float16`、`torch.bfloat16`、`torch.float32`。
- `norm_type`：`["layer", "rms"]`。
- `affine_mode`：`["D", "NAT"]`。
- `scale/shift` 布局：`["BSD", "1", "1SD", "BD", "B1D", "D", "1D",
  "11D", "BF1D"]`。
- `D % 256 == 0` 且 `D <= 8192`。
- `eps`：`1e-5`。
- 容差：非 fp32 使用 `atol=5e-2`、`rtol=5e-2`；fp32 使用 `1e-5`。

Oracle：Layer/RMS norm 后执行 `* tanh(scale) + shift`。第二个 norm-scale 变体在
同一网格上增加 `weight2`、`bias2` 和 `scale2`。

## CuTe DSL Norm Scale Shift

任务：

- `b200_diffusion_cutedsl_norm_scale_shift__multi_shape`
- `h200_diffusion_cutedsl_norm_scale_shift__multi_shape`
- `b200_ltx2_rms_adaln__bitwise`

标准来源：
`python/sglang/jit_kernel/tests/diffusion/test_fused_norm_scale_shift.py`。

回归网格：

- `SHAPES = [(B, S, F, D)]`：`[(1, 1024, 8, 3072), (4, 512, 16, 3072)]`。
- `dtype`：`torch.float16`、`torch.bfloat16`、`torch.float32`。
- `norm_type`：`["layer", "rms"]`。
- `affine_mode`：`["D", "NAT"]`。
- `scale/shift` 布局：`["BSD", "1", "1SD", "BD", "B1D", "D", "1D",
  "11D", "BF1D"]`。
- `BF1D` 布局要求 `S % F == 0`；为 frame 不能整除的情况添加一条拒绝验证行。
- `D % 256 == 0` 且 `D <= 8192`。
- `eps`：`1e-5`。
- 容差：非 fp32 使用 `atol=5e-2`、`rtol=5e-2`；fp32 使用 `1e-5`。

Oracle：`norm(x) * (1 + scale) + shift`，scale/shift 按上述布局进行广播。residual
变体还会使用 `residual` 和 `gate`，并同时输出 `y` 和 `res_out`。

对于 `b200_ltx2_rms_adaln__bitwise`，不允许使用容差。候选必须与任务本地
PyTorch eager 基线
`torch.nn.functional.rms_norm(x, (D,), eps=eps) * (1 + scale) + shift`
位级相等。使用 `torch.equal` 比较最终输出。
