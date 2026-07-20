# 扩散模型基准测试 Shape 覆盖

[English](diffusion_benchmark_shape_coverage.md) | 简体中文

本文将独立扩散模型 Kernel 任务映射到以下文档中的 SGLang 扩散模型基准测试/性能
分析 preset：

`/Users/bbuf/工作目录/Common/sglang/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/benchmark-and-profile.md`

当前辅助脚本定义了 20 个 preset：

`flux`、`flux2`、`qwen`、`qwen-edit`、`zimage`、`wan-t2v`、`wan-ti2v`、
`ltx2`、`ltx23-ti2v-two-stage`、`wan-i2v`、`ltx23-one-stage`、
`ltx23-two-stage`、`ltx23-two-stage-cfg-parallel`、`hunyuanvideo`、
`mova-720p`、`helios`、`joyai-edit`、`firered-edit-1.0`、
`firered-edit-1.1`、`hunyuan3d-shape`。

以下各行是 reset 前保留的 shape 捕获与 2026-06-03 新运行的 B200/H200 捕获的并集。
对于这些 Kernel，Tensor shape 与架构无关，但每个 B200/H200 任务仍必须在自己的目标
GPU 上测试。只有当 `bench/workloads.json` 包含以下所有相关行，或者为当前未调用目标
入口的 preset 记录实时 no-call 证明时，工作负载才算完整。

## 新捕获审计：2026-06-03

有效的原生 SGLang 捕获：

- B200（`ion-b200`）上的 `zimage`：原生运行完成，denoise `0.49s`，产生 17 条
  Kernel 调用记录。捕获了 RMSNorm、QKNorm+RoPE 和 Z-Image
  norm/tanh/mul/add shape。
- H200（`ion8-h200`）上的 `ltx23-one-stage`：原生运行完成，denoise
  `87.11s`，产生 90 条 `ltx2_rotary.apply_ltx2_split_rotary_emb` 记录。
- H200（`ion8-h200`）上的 `firered-edit-1.1`：原生运行完成，denoise
  `24.17s`，产生 64 条 QKNorm+RoPE、CuTe DSL norm/scale/shift 和 Triton
  scale-shift 目标 Kernel 记录。
- H200（`ion8-h200`）上的 `helios`：原生运行完成，denoise `80.04s`，产生 20 条
  LayerNorm 和 `scale_residual_norm_scale_shift.fused_norm_scale_shift` 目标
  Kernel 记录。

以下运行无效或被阻塞，不能用作 shape 覆盖：

- `flux` 和 `flux2`：访问 `black-forest-labs/FLUX.*` 时都因 Hugging Face `403`
  gated-repo 权限失败；日志还触发了原生后端门禁
  （`model fell back to the diffusers backend`），因此没有有效的 shape 行。
- `ltx23-two-stage-cfg-parallel`：H200 运行进入原生加载，但加载 LTX-2.3 两阶段
  Transformer 时发生 H200 OOM，在 denoise 前失败。随后尝试使用显存更大的 B200，
  但那里没有缓存精确的 `Lightricks/LTX-2.3` snapshot，下载超过十分钟仍未完成，
  因此重试在只有安装记录时被停止。
- `hunyuan3d-shape`：H200 原生运行进入 `Hunyuan3DShapeDenoisingStage`，但在
  step 0 因 `_predict_noise_with_cfg() got an unexpected keyword argument 'cfg_policy'`
  失败。随后在另一个 SGLang 提交上使用 B200 重试，数分钟后仍未出现
  任何目标 Kernel 记录，因而停止。应将该 preset 视为运行时阻塞，而不是 no-call 证明。

## 新捕获审计：2026-06-28 LTX2 位级精确任务

有效的原生 SGLang 捕获：

- B200（`ion-b200`）上的 `Lightricks/LTX-2.3`，容器
  `sglang_bbuf_pr29315`，worktree
  `/home/sglang-omni/bbuf/tmp/ltx2_shape_capture_main`，提交
  `828411e6f1`（`origin/main`）。命令形式：
  `CUDA_VISIBLE_DEVICES=5,6 HF_HUB_OFFLINE=1 LTX2_SHAPE_CAPTURE_PATH=...`
  `sglang generate --model-path Lightricks/LTX-2.3 --prompt "A cat and a dog baking a cake together in a kitchen."`
  `--width 768 --height 512 --num-frames 121 --seed 42 --num-gpus 2`
  `--cfg-parallel-size 2 --pipeline-class-name LTX2TwoStagePipeline`
  `--ltx2-two-stage-device-mode original --num-inference-steps 1`
  `--warmup false --enable-torch-compile false --no-save-output`。
  运行完成并产生 42 条目标记录：18 条 `rms_adaln`、12 条
  `qknorm_split_rope_pair`、6 条 `dual_modulate` 和 6 条
  `ca_dual_modulate_from_temb`。

被阻塞的运行：

- 相同 B200 环境上的 `Lightricks/LTX-2` 在 denoise 前失败，因为本地 Hugging Face
  cache snapshot `47da56e2ad66ce4125a9922b4a8826bf407f9d0a` 不完整：存在
  Transformer 索引，但缺少全部 8 个 Transformer safetensors shard。不要把该失败
  运行当作 shape 覆盖。

## `b200_ltx2_rms_adaln__bitwise`

入口：

- `sglang.multimodal_gen.runtime.models.dits.ltx_2:_ltx2_rms_adaln`
- `sglang.multimodal_gen.runtime.models.dits.ltx_2:_ltx2_try_fused_rms_adaln`

2026-06-28 LTX2.3 捕获要求的实时 shape 行：

- 第一阶段视频 self-attention / prompt-Q / MLP：
  `x=[2,1536,4096]`，`scale/shift=[2,1536,4096]`，bf16 contiguous，
  `eps=1e-6`。
- 第一阶段音频 self-attention / prompt-Q / MLP：
  `x=[2,126,2048]`，`scale/shift=[2,126,2048]`，bf16 contiguous，
  `eps=1e-6`。
- 第二阶段视频 self-attention / prompt-Q / MLP：
  `x=[1,6144,4096]`，`scale/shift=[1,6144,4096]`，bf16 contiguous，
  `eps=1e-6`。
- 第二阶段音频 self-attention / prompt-Q / MLP：
  `x=[1,126,2048]`，`scale/shift=[1,126,2048]`，bf16 contiguous，
  `eps=1e-6`。

## `b200_ltx2_dual_modulate__bitwise`

入口：

- `sglang.multimodal_gen.runtime.models.dits.ltx_2:_ltx2_try_fused_rmsnorm_dual_modulate`
- `sglang.multimodal_gen.runtime.models.dits.ltx_2:_ltx2_try_fused_rmsnorm_ca_dual_modulate`

2026-06-28 LTX2.3 捕获要求的实时显式双调制行：

- 第一阶段视频 AV cross-attention：`x=[2,1536,4096]`，
  `scale0/shift0/scale1/shift1=[2,1,4096]`，bf16，`eps=1e-6`。
- 第一阶段音频 AV cross-attention：`x=[2,126,2048]`，
  `scale0/shift0/scale1/shift1=[2,1,2048]`，bf16，`eps=1e-6`。
- 第二阶段视频 AV cross-attention：`x=[1,6144,4096]`，
  `scale0/shift0/scale1/shift1=[1,1,4096]`，bf16，`eps=1e-6`。
- 第二阶段音频 AV cross-attention：`x=[1,126,2048]`，
  `scale0/shift0/scale1/shift1=[1,1,2048]`，bf16，`eps=1e-6`。

要求的实时 timestep-to-cross-attention 行：

- 第一阶段视频 AV cross-attention：`x=[2,1536,4096]`，
  `temb_scale_shift=[2,1,16384]`，`scale_shift_table=[4,4096]` bf16，
  `eps=1e-6`。
- 第一阶段音频 AV cross-attention：`x=[2,126,2048]`，
  `temb_scale_shift=[2,1,8192]`，`scale_shift_table=[4,2048]` bf16，
  `eps=1e-6`。
- 第二阶段视频 AV cross-attention：`x=[1,6144,4096]`，
  `temb_scale_shift=[1,1,16384]`，`scale_shift_table=[4,4096]` bf16，
  `eps=1e-6`。
- 第二阶段音频 AV cross-attention：`x=[1,126,2048]`，
  `temb_scale_shift=[1,1,8192]`，`scale_shift_table=[4,2048]` bf16，
  `eps=1e-6`。

## `b200_ltx2_qknorm_split_rope__bitwise`

入口：

- `sglang.multimodal_gen.runtime.models.dits.ltx_2:_ltx2_try_fused_qknorm_split_rope`
- `sglang.multimodal_gen.runtime.models.dits.ltx_2:apply_split_rotary_emb`

以下是 2026-06-28 LTX2.3 捕获要求的实时 shape 行。所有行都使用 bf16 contiguous
`q/k`，以及 last-dim stride 为 1 的 bf16 non-contiguous split-RoPE cos/sin Tensor；
`num_heads=32`、`eps=1e-6`。

- 第一阶段视频 self-attention：`q/k=[2,1536,4096]`，`head_dim=128`，
  `cos/sin=[2,32,1536,64]`。
- 第一阶段音频 self-attention：`q/k=[2,126,2048]`，`head_dim=64`，
  `cos/sin=[2,32,126,32]`。
- 第一阶段 audio-to-video cross-attention：`q=[2,1536,2048]`，
  `k=[2,126,2048]`，`head_dim=64`，`q cos/sin=[2,32,1536,32]`，
  `k cos/sin=[2,32,126,32]`。
- 第一阶段 video-to-audio cross-attention：`q=[2,126,2048]`，
  `k=[2,1536,2048]`，`head_dim=64`，`q cos/sin=[2,32,126,32]`，
  `k cos/sin=[2,32,1536,32]`。
- 第二阶段视频 self-attention：`q/k=[1,6144,4096]`，`head_dim=128`，
  `cos/sin=[1,32,6144,64]`。
- 第二阶段音频 self-attention：`q/k=[1,126,2048]`，`head_dim=64`，
  `cos/sin=[1,32,126,32]`。
- 第二阶段 audio-to-video cross-attention：`q=[1,6144,2048]`，
  `k=[1,126,2048]`，`head_dim=64`，`q cos/sin=[1,32,6144,32]`，
  `k cos/sin=[1,32,126,32]`。
- 第二阶段 video-to-audio cross-attention：`q=[1,126,2048]`，
  `k=[1,6144,2048]`，`head_dim=64`，`q cos/sin=[1,32,126,32]`，
  `k cos/sin=[1,32,6144,32]`。

## 待审计的已知缺口

- `firered-edit-1.1`、`ltx23-one-stage`、`helios` 和 `zimage` 已于
  2026-06-03 成功重新捕获，相关记录已合并到下文各节。
- 在当前辅助工具中，`helios` 是 QKNorm+RoPE 和 Triton scale-shift 的实时 no-call
  证明：原生运行完成，但只产生 `norm.norm_infer` 和
  `scale_residual_norm_scale_shift` 目标记录。
- `firered-edit-1.1` 是 `norm.norm_infer`、`rmsnorm_onepass` 和
  `norm_tanh_mul_add*` 的实时 no-call 证明：原生运行完成，但只产生 QKNorm+RoPE、
  CuTe DSL norm/scale/shift 和 Triton scale-shift 目标记录。
- `flux` 和 `flux2` 仍被 gated Hugging Face 访问阻塞。在使用已授权 token 和原生
  SGLang 后端日志重新运行之前，不要将其 shape 标记为完整。
- `ltx23-two-stage-cfg-parallel` 在 2026-06-03 审计中被阻塞，但上述
  2026-06-28 B200 LTX2.3 捕获已经完成三个 LTX2 位级精确任务入口。该 preset 中
  任何其他目标入口仍需要自己的实时捕获。
- `hunyuan3d-shape` 仍被当前 SGLang 运行时行为阻塞。不要将失败运行视作 no-call
  证据。

## `diffusion_qknorm_rope__multi_shape`

入口：`qknorm_rope.fused_inplace_qknorm_rope`。

要求保留的实时 shape 行：

- `qwen`：`q/k=[19,24,128]`、`[47,24,128]`、`[4096,24,128]` bf16；
  `cos_sin_cache=[S,128]` fp32；`positions=[S]` int64；`eps=1e-6`。
- `qwen-edit`：`q/k=[189,24,128]`、`[195,24,128]`、`[8424,24,128]` bf16；
  `cos_sin_cache=[S,128]` fp32；`positions=[S]` int64；`eps=1e-6`。
- `zimage`：`q/k=[32,30,128]`、`[4096,30,128]`、`[4128,30,128]` bf16；
  `cos_sin_cache=[S,128]` fp32；`positions=[S]` int64；`eps=1e-5`。
- `joyai-edit`：`q/k=[7904,32,128]` bf16；`cos_sin_cache=[7904,128]` fp32；
  `positions=[7904]` int64；`eps=1e-6`。
- `firered-edit-1.1`：`q/k=[189,24,128]`、`[195,24,128]`、
  `[8424,24,128]` bf16；`cos_sin_cache=[S,128]` fp32；`positions=[S]` int64；
  `eps=1e-6`。

当前 preset 审计状态：`helios` 是实时 no-call；`firered-edit-1.1` 已捕获；
`flux` 和 `flux2` 被 gated 访问阻塞；`hunyuan3d-shape` 被运行时阻塞。除非重新
捕获，`firered-edit-1.0` 仍只有保留记录覆盖。

## `diffusion_norm_infer__multi_shape`

入口：

- `norm.norm_infer`
- `rmsnorm_onepass.triton_one_pass_rms_norm`

要求保留的实时 shape 行：

- `helios` LayerNorm：`x=[8640,5120]` fp32，`weight=[5120]` fp32，
  `bias=[5120]` fp32，`eps=1e-6`，`is_rms_norm=False`。
- `zimage` RMSNorm：`x=[4096,128]` 和 `[16384,128]` bf16，
  `weight=[128]` bf16，`eps=1e-6`。
- `hunyuanvideo` RMSNorm：`x=[1320,128]`、`[648720,128]`、
  `[650040,128]` bf16，`weight=[128]` bf16，`eps=1e-6`。

当前 preset 审计状态：`firered-edit-1.1` 对两个入口都是实时 no-call；`flux2`
被 gated 访问阻塞；`hunyuan3d-shape` 被运行时阻塞。

## `diffusion_cutedsl_norm_tanh_mul_add__multi_shape`

入口：

- `norm_tanh_mul_add_norm_scale.fused_norm_tanh_mul_add`
- `norm_tanh_mul_add_norm_scale.fused_norm_tanh_mul_add_norm_scale`

要求保留的实时 shape 行：

- `zimage`：`hidden=[1,4096,3840]` 和 `[1,4128,3840]` bf16，norm weight
  `[3840]` bf16，modulation `[1,1,3840]` bf16，residual/output
  `[1,S,3840]` bf16，`norm_type=rms`，`eps=1e-5`。

当前 preset 审计状态：`firered-edit-1.1`、`helios` 和 `ltx23-one-stage`
对这两个入口都是实时 no-call；`hunyuan3d-shape` 被运行时阻塞。

## `diffusion_cutedsl_norm_scale_shift__multi_shape`

入口：

- `scale_residual_norm_scale_shift.fused_norm_scale_shift`
- `scale_residual_norm_scale_shift.fused_scale_residual_norm_scale_shift`

要求保留的实时 shape 行：

- `qwen`：`hidden=[1,S,3072]` bf16，`S=19,47,4096`；scale/shift 为
  `[1,3072]` 或 `[1,1,3072]` bf16。
- `qwen-edit`：`hidden=[1,S,3072]` bf16，`S=189,195`；scale/shift 为
  `[1,3072]` bf16。
- `firered-edit-1.0`：`hidden=[1,8424,3072]` bf16；scale/shift 为
  `[1,1,3072]` bf16。
- `joyai-edit`：`hidden=[1,S,4096]` bf16，`S=997,1004,7904`；scale/shift
  为 `[1,4096]` bf16。
- `hunyuanvideo`：`hidden=[1,S,3072]` bf16，`S=55,27030,27085`；scale/shift
  为 `[1,3072]` bf16。
- `wan-ti2v`：`hidden=[1,18144,3072]` bf16；scale/shift 同时包含 bf16
  `[1,18144,3072]` 和 fp32 `[1,18144,3072]` 变体。
- `wan-t2v`：`hidden=[1,37800,5120]` 和 `[1,75600,5120]` bf16；scale/shift
  同时包含 fp32 `[1,1,5120]` 和 bf16 `[1,1,5120]` 变体。
- `wan-i2v`：`hidden=[1,37044,5120]` 和 `[1,74088,5120]` bf16；scale/shift
  同时包含 fp32 `[1,1,5120]` 和 bf16 `[1,1,5120]` 变体。
- `mova-720p`：`hidden=[1,101,1536]`、`[1,44100,5120]`、
  `[1,176400,5120]` bf16。
- `helios` 的 `fused_norm_scale_shift`：`hidden=[1,8640,5120]` bf16，搭配
  bf16 full-shape scale/shift；以及 `hidden=[1,11040,5120]` bf16，搭配 fp32
  full-shape scale/shift；residual 参数为 `None`；`norm_type=layer`，
  `eps=1e-6`。
- `firered-edit-1.1`：
  - `fused_norm_scale_shift`：`hidden=[1,S,3072]` bf16，
    `S=189,195,8424`；residual 参数为 `None`；当 `S=189,195` 时 scale/shift 为
    `[1,3072]` bf16，当 `S=8424` 时为 `[1,1,3072]` bf16；
    `norm_type=layer`，`eps=1e-6`。
  - `fused_scale_residual_norm_scale_shift`：`hidden=[1,S,3072]`、
    `residual=[1,S,3072]`，以及 output/residual-output Tensor 在
    `S=189,195,8424` 时均为 bf16；pre-scale 为 `[1,1,3072]` bf16；当
    `S=189,195` 时 scale/shift 为 `[1,3072]` bf16，当 `S=8424` 时为
    `[1,1,3072]` bf16；`norm_type=layer`，`eps=1e-6`。

对于 residual 入口，每一行都要包含复制的 SGLang 基线 ABI 所展示的对应 residual
Tensor。

当前 preset 审计状态：`firered-edit-1.1` 已捕获两个入口；`helios` 只捕获了
`fused_norm_scale_shift`；`hunyuan3d-shape` 被运行时阻塞。本文档之后新增的任何
preset 仍必须重新进行实时审计。

## `diffusion_fuse_scale_shift__multi_shape`

入口：

- `scale_shift.fuse_scale_shift_kernel`
- `scale_shift.fuse_layernorm_scale_shift_gate_select01_kernel`
- `scale_shift.fuse_residual_layernorm_scale_shift_gate_select01_kernel`

要求保留的实时 shape 行：

- `qwen`：`x=[1,S,3072]` bf16，`S=19,47,4096`；scale/shift 为 `[1,1,3072]`
  bf16；`scale_constant=0`。
- `qwen-edit`：`x=[1,S,3072]` bf16，`S=189,195,8424`；
  `fuse_scale_shift_kernel` 同时包含 `[1,1,3072]` 和 full-shape
  `[1,8424,3072]` scale/shift 用例。
- `qwen-edit` gated 变体：`x=[1,8424,3072]` bf16，`index=[1,8424]` int32，
  `scale{0,1}/shift{0,1}/gate{0,1}=[1,3072]` bf16；residual 变体还包含
  `residual=[1,8424,3072]` 和 `residual_gate=[1,8424,3072]` bf16。
- `firered-edit-1.0`：`x=[1,8424,3072]` bf16，scale/shift `[1,1,3072]`
  bf16。
- `firered-edit-1.1`：`x=[1,S,3072]` bf16，`S=189,195,8424`；scale 为
  `[1,1,3072]` bf16，output 为 `[1,S,3072]` bf16，`scale_constant=0`。
- `hunyuanvideo`：`x=[1,S,3072]` bf16，`S=55,27030,27085`；scale/shift 为
  `[1,3072]` bf16。
- `wan-ti2v`：`x=[1,18144,3072]` bf16，scale/shift 为
  `[1,18144,3072]` fp32 non-contiguous。
- `wan-t2v`：`x=[1,37800,5120]` bf16，scale/shift `[1,1,5120]` fp32。
- `wan-i2v`：`x=[1,37044,5120]` bf16，scale/shift `[1,1,5120]` fp32。

当前 preset 审计状态：`helios` 是实时 no-call；`firered-edit-1.1` 已捕获；
`joyai-edit` 仍只有保留记录，除非重新捕获；`hunyuan3d-shape` 被运行时阻塞。

## `diffusion_rotary_embedding__multi_shape`

入口：

- `rotary.apply_rotary_embedding`
- `ltx2_rotary.apply_ltx2_split_rotary_emb`

要求保留的实时 shape 行：

- `hunyuanvideo` 标准 RoPE：`x=[1,27030,24,128]` bf16，`cos=[27030,64]`
  fp32，`sin=[27030,64]` fp32，`interleaved=False`。
- `ltx2`：split RoPE 行：
  `[1,126,2048]` 搭配 cos/sin `[1,32,126,32]` bf16 non-contiguous；
  `[1,1536,2048]` 搭配 `[1,32,1536,32]`；
  `[1,1536,4096]` 搭配 `[1,32,1536,64]`；
  `[1,6144,2048]` 搭配 `[1,32,6144,32]`；
  `[1,6144,4096]` 搭配 `[1,32,6144,64]`。
- `ltx23-ti2v-two-stage`：与 `ltx2` 相同的单 batch split 行，`S=126,1536,6144`，
  hidden size 为 `2048/4096`。
- `ltx23-one-stage`：2026-06-03 实时捕获的 split RoPE 行：
  `[1,126,2048]` 搭配 cos/sin `[1,32,126,32]` bf16 non-contiguous，
  `[1,6144,2048]` 搭配 `[1,32,6144,32]`，以及
  `[1,6144,4096]` 搭配 `[1,32,6144,64]`。
- `ltx23-two-stage`：高分辨率 split 行：
  `[2,126,2048]`、`[2,6144,2048]`、`[2,6144,4096]`、
  `[1,24576,2048]`、`[1,24576,4096]`，搭配对应的 non-contiguous
  `[B,32,S,32/64]` cos/sin Tensor。

当前 preset 审计状态：`ltx23-one-stage` 已捕获；`ltx23-two-stage-cfg-parallel`
仍被 H200 OOM 和 B200 缓存缺失阻塞；`hunyuan3d-shape` 被运行时阻塞。

## `diffusion_group_norm_silu__multi_shape`

入口：

- `group_norm_silu.apply_group_norm_silu`
- `group_norm_silu.triton_group_norm_silu`

要求保留的实时 shape 行：

- 仅使用 `hunyuanvideo` VAE shape。
- 数据类型为 fp16。Triton 入口使用 `num_groups=32`、`eps=1e-6`。
- 当目标架构中观察到时，同时包含 contiguous 和 non-contiguous 用例。保留的并集
  覆盖 channel 数 `512`、`256`、`128`，时间深度 `2`、`3`、`5`、`9`、`17`，以及
  包括 `12x10`、`12x32`、`24x20`、`24x64`、`32x10`、`32x32`、`48x40`、
  `48x128`、`64x20`、`64x64`、`96x80`、`96x256`、`128x40`、`128x128`、
  `256x80` 和 `256x256` 在内的空间尺寸对。

由于此系列包含大量行，应从实时 HunyuanVideo 捕获或 reset 前 Git 历史中保留的原始
JSONL 生成 `bench/workloads.json`，不要手写一个缩减列表。

当前 preset 审计状态：`zimage`、`firered-edit-1.1`、`helios` 和 `ltx23-one-stage`
对这些入口都是实时 no-call；`hunyuan3d-shape` 被运行时阻塞。

## `diffusion_causal_conv3d_cat_pad__multi_shape`

入口：

- `causal_conv3d_pad.fused_causal_conv3d_cat_pad`

B200 新捕获审计：2026-06-24，`cosmos3-nano-t2v`，`nvidia/Cosmos3-Nano`，
832x480，9 帧，4 个 denoise step。全阶段 trace 显示 `_fused_cat_pad_5d_kernel`
耗时 11.94 ms。原始捕获：

`/tmp/sglang_profile_b200/outputs/shape_captures/cosmos3-nano-t2v_no_compile_v2.jsonl`

要求的实时 shape 行：

- bf16 contiguous `x=[1,1024,1,30,52]`，`cache=[1,1024,1,30,52]`，
  padding `[1,1,1,1,2,0]`。
- bf16 contiguous `x=[1,1024,1,30,52]`，`cache=[1,1024,2,30,52]`，
  padding `[1,1,1,1,2,0]`。
- bf16 contiguous `x=[1,1024,2,60,104]`，`cache=[1,1024,1,60,104]`，
  padding `[1,1,1,1,2,0]`。
- bf16 contiguous `x=[1,1024,2,60,104]`，`cache=[1,1024,2,60,104]`，
  padding `[1,1,1,1,2,0]`。
- bf16 contiguous `x=[1,512,4,120,208]`，`cache=[1,512,1,120,208]`，
  padding `[1,1,1,1,2,0]`。
- bf16 contiguous `x=[1,512,4,120,208]`，`cache=[1,512,2,120,208]`，
  padding `[1,1,1,1,2,0]`。
- bf16 contiguous `x=[1,256,4,240,416]`，`cache=[1,256,1,240,416]`，
  padding `[1,1,1,1,2,0]`。
- bf16 contiguous `x=[1,256,4,240,416]`，`cache=[1,256,2,240,416]`，
  padding `[1,1,1,1,2,0]`。

将低 count 的 cache-null、no-pad 和已捕获 non-contiguous 行作为回归行加入
`bench/workloads.json`。

## `diffusion_attention_concat_copy__multi_model`

入口源码模式：

- `USPAttention._forward_with_replicated_prefix`
- `USPAttention._forward_with_replicated_kv_prefix_split`
- `runtime/layers/attention/layer.py` 中本地 `contiguous()` 加
  `torch.cat(..., dim=1)` 的 attention 布局模式。

B200 新捕获审计：2026-06-24，保留自 JoyAI Image Edit 和 FLUX.2 Klein Base 的
torch profiler shape。

要求的实时 shape 行：

- `concat_sequence`：bf16 `a=[1,512,24,128]`、`b=[1,4096,24,128]`，
  output `[1,4608,24,128]`。
- `concat_sequence`：bf16 `a=[1,8048,32,128]`、`b=[1,1004,32,128]`，
  output `[1,9052,32,128]`。
- `copy_contiguous`：bf16 `[1,4608,24,128]`。
- `copy_contiguous`：bf16 `[1,8048,32,128]`。
- `copy_contiguous`：bf16 `[1,1004,32,128]`。
- `slice_heads_then_concat_sequence`：与 FLUX.2 匹配的 bf16 prefix/shard 行，
  prefix 长度 512、shard 长度 4096、24 个 head、head dim 128。
- `slice_heads_then_concat_sequence`：与 JoyAI 匹配的 bf16 prefix/shard 行，
  长度 1004 和 8048、32 个 head、head dim 128。

当前证据：JoyAI trace 显示 `CatArrayBatchedCopy` 耗时 521.5 ms，以及较大的
contiguous/copy 行；FLUX.2 Klein Base 在 replicated-prefix attention 路径中显示
重复的 copy/cat 行。

## `diffusion_residual_gate_add__multi_shape`

入口源码模式：

- `LTX2TransformerBlock.forward`
- `Ideogram4TransformerBlock.forward`
- `flux_2.py` 中 FLUX.2 modulation 和 residual gate 表达式。

B200 新捕获审计：2026-06-24，保留自 LTX-2.3 HQ、Ideogram4 FP8 和 FLUX.2 Klein
Base 的 torch profiler shape。

要求的实时 shape 行：

- `residual_gate_add`：bf16 `residual=[1,8160,4096]`、
  `update=[1,8160,4096]`、`gate=[1,8160,4096]`。
- `residual_gate_add`：bf16 `residual=[1,32640,4096]`、
  `update=[1,32640,4096]`、`gate=[1,1,4096]`。
- `residual_gate_add`：bf16 `residual=[1,126,2048]`、
  `update=[1,126,2048]`、`gate=[1,126,2048]`。
- `residual_gate_add`：bf16 `residual=[1,4096,4608]`、
  `update=[1,4096,4608]`、`gate=[1,1,4608]`。
- `residual_gate_add`：bf16 `residual=[1,4608,3072]`、
  `update=[1,4608,3072]`、`gate=[1,1,3072]`。
- `residual_gate_add`：bf16 `residual=[1,4096,3072]`、
  `update=[1,4096,3072]`、`gate=[1,1,3072]`。
- `residual_gate_add`：bf16 `residual=[1,512,3072]`、
  `update=[1,512,3072]`、`gate=[1,1,3072]`。
- `broadcast_add_4d`：bf16 `a=[1,1,3,2048]`、`b=[1,126,3,2048]`，
  output `[1,126,3,2048]`。

当前证据：LTX-2.3 HQ trace 显示所有 CUDA Kernel 中逐元素 add 耗时 7260.2 ms、
mul 耗时 4022.4 ms。这些行属于内存带宽任务；GEMM、attention、QKNorm+RoPE 以及
已有的 norm/scale/shift Kernel 不属于本系列范围。
