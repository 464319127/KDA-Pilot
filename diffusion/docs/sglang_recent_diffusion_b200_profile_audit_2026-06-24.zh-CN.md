# SGLang 近期扩散模型 B200 性能分析审计：2026-06-24

[English](sglang_recent_diffusion_b200_profile_audit_2026-06-24.md) | 简体中文

本审计记录近期 SGLang 扩散模型支持在 B200 上的性能分析证据，以及收集真实生产
shape 后看起来值得行动的 KDA 任务边界。

## 来源与机器

- SGLang worktree：`/tmp/sglang_profile_b200`
- SGLang 提交：`84a7a8401842b47b9e372df8136b7b7dc0b7e166`
- 主机/容器：`ion-b200` / `sglang_bbuf`
- GPU：NVIDIA B200
- 原生后端门禁：所有保留的基准测试/性能分析行都来自 SGLang 原生扩散模型路径。

## 已分析模型

| 模型 preset | 模型 | Shape | Denoise / E2E | 峰值预留显存 | 证据路径 |
| --- | --- | --- | --- | --- | --- |
| `cosmos3-nano-t2v` | `nvidia/Cosmos3-Nano` | 832x480，9 帧，4 步 | 169.55 ms / 216.8 ms | 35,088 MB | `/tmp/sglang_profile_b200/outputs/diffusion_profiles/cosmos3-nano-t2v_nocompile_all/bfc39a58-f657-4c3c-86a4-728c58484da6-full_stages-global-rank0.trace.json.gz` |
| `ideogram4-fp8` | `ideogram-ai/ideogram-4-fp8` | 1024x1024，20 步 | 5592.9 ms / 5713.1 ms | 33,186 MB | `/tmp/sglang_profile_b200/outputs/diffusion_profiles/ideogram4-fp8_nocompile_all/bf5ca92b-9bc5-41a8-bf2a-8f2db371b9d4-full_stages-global-rank0.trace.json.gz` |
| `sana-1.5-1.6b` | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` | 1024x1024 | 456.9 ms / 547.0 ms | 9856 MB | `/tmp/sglang_profile_b200/outputs/diffusion_profiles/sana-1.5-1.6b_nocompile_all/3d8259f6-4b7c-4647-b0f4-3847332c116f-full_stages-global-rank0.trace.json.gz` |
| `flux2-klein-base` | `black-forest-labs/FLUX.2-klein-base-4B` | 1024x1024，50 步 | 3709.1 ms / 3856.4 ms | 13,442 MB | `/tmp/sglang_profile_b200/outputs/diffusion_profiles/flux2-klein-base_nocompile_all/df2cd86f-354e-4576-a41a-6d428f974685-full_stages-global-rank0.trace.json.gz` |
| `joyai-edit` | `jdopensource/JoyAI-Image-Edit-Diffusers` | 1024x1024，40 步，CFG 并行 | 7.27 s / 8.23 s | 41,976 MB | `/tmp/sglang_profile_b200/outputs/diffusion_profiles/joyai-edit_nocompile_all/f2159a93-363e-4c7c-8bd4-4a93017b3985-full_stages-global-rank0.trace.json.gz` |
| `ltx23-hq-two-stage` | `Lightricks/LTX-2.3` | 1920x1088，121 帧 | 42.34 s / 55.29 s | 67,902 MB | `/tmp/sglang_profile_b200/outputs/diffusion_profiles/ltx23-hq-two-stage_nocompile_all/70769941-6405-49c2-8663-a632325a001b-full_stages-global-rank0.trace.json.gz` |

## 访问说明

- 当前环境无法通过 Hugging Face 直连或 HF 镜像访问 `krea/Krea-2`。
- 可以通过 Hugging Face 和 ModelScope 获取 LingBot 模型元数据，但原生路径需要图像/
  动作输入，且没有基准测试辅助 preset。
- 可以通过 ModelScope 获取 SANA-WM 元数据，但完整仓库非常大，并且被分析的代码树中
  没有基准测试辅助 preset。
- `flux2-klein-base` 可以通过手动原生 `sglang generate` 命令访问和运行，不过由于
  FLUX 系列被标记为 gated，辅助工具采用了较保守的策略。

## 可执行的 KDA 任务边界

### Attention concat/copy 布局

真实证据：

- JoyAI profile：`[1,8048,32,128]` 上的 `aten::contiguous` 在 3200 次调用中耗时
  748.6 ms，相应的 `aten::copy_` 耗时 705.1 ms。
  `[[1,8048,32,128], [1,1004,32,128]]` 上的 `aten::cat` 在 4800 次调用中耗时
  71.6 ms。GPU trace 还显示 `CatArrayBatchedCopy` 耗时 521.5 ms。
- FLUX.2 Klein Base profile：`[1,4608,24,128]` 上的 `aten::copy_` 在 4000 次调用中
  耗时 154.9 ms。`[[1,512,24,128], [1,4096,24,128]]` 上的 `aten::cat` 在
  1500 次调用中耗时 51.2 ms。
- 源码边界：`python/sglang/multimodal_gen/runtime/layers/attention/layer.py`
  的 `USPAttention._forward_with_replicated_prefix` 和
  `_forward_with_replicated_kv_prefix_split` 中存在重复的 `contiguous()` 加
  `torch.cat` 路径。

KDA 任务：`b200_diffusion_attention_concat_copy__multi_model`。

### Residual gate add 和广播加法

真实证据：

- LTX-2.3 HQ profile：聚合 CUDA Kernel 主要由 DiT block 周围的逐元素 add/mul 主导
  （`aten::add` 和 `aten::mul` shape 包括 `[1,8160,4096]`、`[1,32640,4096]` 和
  `[1,126,2048]`）。CPU 算子 shape 汇总包含 `[1,8160,4096]` 上的 27,927 次 add、
  25,143 次 mul，以及 `[1,126,2048]` 上的 33,123 次 add、26,877 次 mul。
- LTX 还有一个非常热的广播加法 shape：
  `[1,1,3,2048] + [1,126,3,2048]`，共 13,392 次调用，CPU 算子时间 3139.4 ms。
- Ideogram4 FP8 profile：`[1,1,4608] * [1,4096,4608]` 上的 `aten::mul` 在
  1360 次调用中耗时 150.7 ms；周围的 `x + gate * norm(...)` 源码位于
  `runtime/models/dits/ideogram.py`。
- FLUX.2 Klein Base profile：重复的 gate/modulation shape 包括
  `[1,1,3072] * [1,4608,3072]`、`[1,1,3072] * [1,4096,3072]` 和
  `[1,1,3072] * [1,512,3072]`。

KDA 任务：`b200_diffusion_residual_gate_add__multi_shape`。

### Causal Conv3D cat/pad

真实证据：

- Cosmos3 Nano T2V 全阶段 profile 显示 VAE/decode 路径中的
  `_fused_cat_pad_5d_kernel` 总耗时 11.94 ms。
- Shape 捕获路径：
  `/tmp/sglang_profile_b200/outputs/shape_captures/cosmos3-nano-t2v_no_compile_v2.jsonl`
- 代表性 bf16 工作负载包括：
  - `x=[1,1024,1,30,52]`，`cache=[1,1024,1,30,52]`，padding `[1,1,1,1,2,0]`
  - `x=[1,1024,2,60,104]`，`cache=[1,1024,1,60,104]`，padding `[1,1,1,1,2,0]`
  - `x=[1,512,4,120,208]`，`cache=[1,512,1,120,208]`，padding `[1,1,1,1,2,0]`
  - `x=[1,256,4,240,416]`，`cache=[1,256,1,240,416]`，padding `[1,1,1,1,2,0]`

KDA 任务：`b200_diffusion_causal_conv3d_cat_pad__multi_shape`。

## No-Go 或已有任务说明

- QKNorm+RoPE 已由现有 KDA 扩散模型任务覆盖。新的 profile 确认 FLUX.2、JoyAI 和
  Cosmos3 中存在调用，但其独立耗时并不是剩余机会中最大的。
- LTX2 split RoPE 已由现有 rotary 任务覆盖；LTX-2.3 HQ trace 显示
  `_ltx2_split_rotary_kernel` 确实有成本，但低于周围的逐元素 block 工作。
- SANA 1.5 1.6B 在 B200 上已经很快，没有暴露出一个预期收益足够大的新独立
  CUDA 任务。
