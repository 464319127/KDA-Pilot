<div align="center">

# KDA-Pilot

[English](README.md) | 简体中文

**面向 SGLang、证据优先的自主 GPU Kernel 优化项目。**

KDA-Pilot 将真实服务框架中的 Kernel 转化为可复现的优化任务：冻结的生产
shape、复制到本地的上游基线、对称的基准测试、正确性门禁、Nsight Compute
证据、KernelWiki 参考资料，以及集中在一处的 RLCR 风格 Agent 迭代。

[![GitHub stars](https://img.shields.io/github/stars/BBuf/KDA-Pilot?style=social)](https://github.com/BBuf/KDA-Pilot/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/BBuf/KDA-Pilot?style=social)](https://github.com/BBuf/KDA-Pilot/forks)
[![Last commit](https://img.shields.io/github/last-commit/BBuf/KDA-Pilot?style=flat-square)](https://github.com/BBuf/KDA-Pilot/commits/main)
[![B200 diffusion](https://img.shields.io/badge/B200_diffusion-10_tracked_tasks-2ea44f?style=flat-square)](#b200-扩散模型结果)
[![AI Infra Skills](https://img.shields.io/badge/sibling-AI--Infra--Auto--Driven--SKILLS-2f80ed?style=flat-square)](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)
[![Kernel Design Agents](https://img.shields.io/badge/built_on-Kernel--Design--Agents-ff6f00?style=flat-square)](https://github.com/mit-han-lab/kernel-design-agents)

</div>

大多数 AI Kernel 演示只优化一小段代码。KDA-Pilot 优化真正出现在 SGLang
扩散模型和 LLM 服务工作流中的部分，并保留判断 Agent 是否确实改善了生产路径所需的
证据。

如果你关注能够复现、审查，并能与真实框架基线对比的自主
CUDA/Triton/CuTe-DSL 优化，这个仓库值得关注。

## 为什么它很重要

- **真实工作负载，而不是玩具 shape。** 扩散模型任务来自 20 个真实的 SGLang
  扩散模型，并被归并为每个 Kernel 对应的多 shape 工作负载。
- **端到端耗时指标。** 核心数据包含 Python、分发、封装、Kernel 启动和
  `cuda.synchronize()` 的开销，而不只是孤立的设备执行时间。
- **不存在投机取巧的奖励路径。** 基线与候选实现使用匹配的本地 ABI；任务运行时
  不会 monkey-patch 或导入 SGLang。
- **知识引导的迭代。** 任务可以利用 `KernelWiki` 和 `ncu-report-skill`，让已有的
  Blackwell/Hopper Kernel 工作成果和 NCU 瓶颈证据成为优化循环的一部分。
- **带审查的 Agent 循环。** 候选实现能否晋升取决于正确性门禁、运行日志和代码审查，
  而不是“某一行数据跑得快”。

## B200 扩散模型结果

以下是拥有稳定生产基线的 B200 任务相对于对应 SGLang/Triton/CuTe-DSL 基线的
结果。测量包含分发和同步开销，因此更接近用户通过公开 Kernel 路径观察到的表现。
标记为 `review` 的行仍然保留，是因为它们后来成为了面向 SGLang 的快速路径；但在
基线变化后，原任务的核心指标也随之改变。

| Kernel 任务 | B200 证据 | 代表性提升 |
| --- | --- | --- |
| `qknorm_rope` | 1.1341x | 大 shape 行 1.145-1.279x |
| `norm_infer` | 1.3523x | RMS 小 shape 1.634-1.641x |
| `rotary_embedding` | 1.4912x | HunyuanVideo 2.087x；LTX2 1.133-1.622x |
| `cutedsl_norm_tanh_mul_add` | 1.4953x | v1 1.602-1.625x |
| `cutedsl_norm_scale_shift` | 1.3201x | Hunyuan 1.388-1.516x；JoyAI 1.477-1.495x |
| `fuse_scale_shift` | 2.7499x | 小型广播行 7.365-7.891x |
| `group_norm_silu` | 2.3118x | 小/中 C 行 1.369-4.982x；NC 行最高 3.648x |
| `attention_concat_copy` | 1.30x | 按 head 切分的复制 1.39-2.61x；24 个工作负载位级精确 |
| `causal_conv3d_cat_pad` | 2.06x | 生产行 1.60-2.45x；通过 B200 位级精确门禁 |
| `residual_gate_add` | review | 旧 eager 几何平均 2.19x 已被取代；SGLang Triton 行的后续结果约为 1.11x |

## KernelWiki 引导的亮点

| Kernel | KernelWiki / 参考资料 | 关键技术 |
| --- | --- | --- |
| `qknorm_rope` | **TensorRT-LLM PR-13052/11869 DiT QKNorm+RoPE；SGLang PR-15141/19059/21440/21654 融合 QKNorm/RoPE；内存受限模式** | 共享 RoPE 暂存、Q/K 复用、仅对大 shape 行使用分阶段路径 |
| `norm_infer` | **KernelWiki memory-bound/vectorized-loads/register-budgeting；vLLM PR-31828 SM100 RMSNorm 可选路径** | Warp-row RMS、分块持久化 RMS、8B/16B 向量路径 |
| `rotary_embedding` | **SGLang PR-24411 LTX2 split RoPE；vLLM PR-21126/30729 FlashInfer RoPE 路由；vectorized-loads** | 128 位向量 I/O、提前计算 cos/sin、LTX2 分块匹配 |
| `cutedsl_norm_tanh_mul_add` | **KernelWiki memory-bound/vectorized-loads/register-budgeting；NCU long-scoreboard 和 launch-bounds 证据** | 提取行不变量计算、launch-bounds 调优、精确 `tanhf` |
| `cutedsl_norm_scale_shift` | **SGLang PR-14717 CuTe-DSL norm/scale/shift 融合；vectorized-loads；register-budgeting** | 按操作数类型分发、16B/32B 向量、两遍方差计算 |
| `fuse_scale_shift` | **SGLang PR-14717 融合 norm/scale/shift 系列；vectorized-loads；cache-policy；memory-bound 模式** | Rowgrid/flatvec/exact-C 路径、缓存提示、单遍归约 |
| `group_norm_silu` | **SGLang PR-22814/23148/23938 GroupNorm+SiLU；memory-bound 模式；vectorized-loads** | 拆分组统计、generation counter、channels-last 转置 |
| `attention_concat_copy` | **SGLang USP attention concat/copy 路径；内存受限的按 head 切分复制；A/A 测试框架验证** | 单次启动的区域复制、带 pitch 的 16B 分块聚集、严格拒绝不支持的布局/设备 |
| `causal_conv3d_cat_pad` | **SGLang causal Conv3D cat/pad Triton 基线；B200 NCU 指令受限证据** | 扁平化分块、16B 向量化存储、感知 stride 的回退、位级精确门禁 |
| `residual_gate_add` | **SGLang residual/gate/add 服务行；基线迁移到 Triton `fuse_scale_shift_kernel`；PR #29361 后续工作** | 单遍 CUDA 融合、固定 GPU 的正确性验证、基线替换后的明确重测说明 |

配套文章记录了基准结果解读、各 Kernel 的优化路径、KernelWiki/参考资料链接，以及与
AKO4X 的对比：
[KDA-Pilot 优化 SGLang Diffusion Kernel](https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/main/large-language-model/sglang/KDA-Pilot%20%E4%BC%98%E5%8C%96%20SGLang%20Diffusion%20Kernel%20%E6%95%88%E6%9E%9C%E4%B8%8E%E7%BB%8F%E9%AA%8C.md)。

## 仓库内容

```text
diffusion/    SGLang 扩散算子 Kernel 任务。
              每个任务包含复制到本地的基线、优化后的方案、基准测试、
              正确性约定、运行日志和结果记录。

llm/          SGLang 自回归模型 Kernel 工作流优化项目。
              在 B200/H200 上运行高优先级模型，测试低/中/高并发，
              分析前向传播，并将占比 >=1% 的 Kernel 接口转化为带有
              独立启动脚本的扁平任务目录。

external/     可选的共享知识子模块。
              KernelWiki/         Blackwell/Hopper Kernel 设计参考资料
              ncu-report-skill/   Nsight Compute 分析/报告辅助工具
```

建议从以下文档开始：

- [`diffusion/README.md`](diffusion/README.md)：独立扩散模型 Kernel 任务和基准规则。
- [`llm/README.md`](llm/README.md)：LLM Kernel 工作流优化项目。
- [`diffusion/docs/standalone_diffusion_benchmark.md`](diffusion/docs/standalone_diffusion_benchmark.md)：
  基线/候选实现的基准测试约定。
- [`diffusion/docs/diffusion_kernel_rules.md`](diffusion/docs/diffusion_kernel_rules.md)：
  正确性、回退和晋升的约束规则。

## 任务生命周期

每个扩散模型 Kernel 任务都遵循相同的结构：

```text
prompt.md       供 Agent 使用的任务卡
config.toml     基准测试/构建的默认配置
baseline/       从上游 SGLang 复制的基线源码
solution/       优化后的候选实现源码
bench/          独立基准测试与正确性测试框架
docs/           运行日志、性能分析说明、来源说明、决策记录
```

最重要的原则是对称性：Agent 必须通过匹配的本地接口、固定工作负载行、预分配输出、
CUDA Event 计时、交错 A/B 采样、严格的正确性检查和完整的来源记录，对复制的基线和
候选实现进行比较。

## 运行任务

需要使用可选知识参考资料时，初始化子模块：

```bash
git submodule update --init --recursive
```

从仓库根目录启动一个任务：

```bash
diffusion/scripts/launch_kernels/k03_b200_diffusion_qknorm_rope__multi_shape.sh
```

常用环境变量：

```bash
KDA_NO_CLAUDE=1                 # 准备 worktree，但不启动 Agent
KDA_BASE_BRANCH=<ref>           # 从指定的已提交 ref 启动
KDA_BASH_BIN=/opt/homebrew/bin/bash
```

启动脚本不支持 macOS 自带的 `/bin/bash` 3.2，因为嵌套的 Humanize/Codex Hook
依赖现代 Bash 行为。

## 我如何驱动 Agent

这些优化项目使用完全自主模式的 Coding Agent 运行。为了保证可复现性，下面给出我在
准备好的任务 worktree 中启动它们的确切方式。

**Claude Code**：与扩散模型启动脚本使用相同的组合（Opus 4.8 + 最大 effort +
自动/绕过权限模式，也就是 Claude Code UI 中的“Effort (Max)”和“Auto mode”开关）：

```bash
claude --permission-mode bypassPermissions --model opus --effort max
```

`diffusion/scripts/launch_kda_kernel_task.sh` 调用的就是这条命令（默认设置为
`CLAUDE_MODEL=opus`、`CLAUDE_EFFORT=max`、
`--permission-mode bypassPermissions`），因此通过脚本启动任务与手动运行 Claude
效果相同。需要在切换模型后继续迭代长期目标时，可以恢复会话：

```bash
claude --resume <session-uuid>
```

**Ultracode 模式**：最高严谨度设置（Claude Code effort 菜单中的
“Effort (Ultracode – xhigh + workflows)”，并搭配 Auto mode）。Ultracode
并不是一个启动参数的 effort 值：`claude --effort ultracode` 会发出警告
（`Unknown --effort value 'ultracode'`）并回退到默认值；有效的 `--effort`
参数级别为 `low, medium, high, xhigh, max`。Ultracode 是
**xhigh effort + 启用动态工作流**的组合，因此 Agent 会以 xhigh 级别进行推理，
并为实质性任务编写多 Agent 工作流。先在 `/config` 中启用一次
**Dynamic workflows** 开关（设置键为 `enableWorkflows`），然后以 xhigh 启动：

```bash
claude --permission-mode bypassPermissions --model opus --effort xhigh
```

也可以使用一条独立、完整的命令：

```bash
claude --permission-mode bypassPermissions --model opus --effort xhigh --settings '{"enableWorkflows": true}'
```

启用工作流后，会话内的 `/effort` 菜单会显示“Ultracode”，选中它即可固定使用
xhigh。若只想让某一条提示词启用该模式，而不是整个会话，可以在该消息中加入关键词
`ultracode`。

**Codex**：完全访问权限，不显示审批提示：

```bash
codex --yolo --sandbox danger-full-access --ask-for-approval never
```

二者均在无沙箱/自动批准模式下运行，因为每个任务都位于隔离且准备好的 worktree 中，
并拥有自己的基准测试和正确性门禁；只有正确性约定与运行日志一致时，Agent 的优化
结果才会被认可。

## 当前优化项目

- **扩散模型 Kernel：** B200 和 H200 任务目录覆盖 QK norm + RoPE、norm inference、
  rotary embedding、融合 scale/shift、group norm + SiLU、attention concat/copy、
  causal Conv3D cat/pad、residual gate/add、CuTe-DSL norm/tanh/mul/add，以及
  CuTe-DSL norm/scale/shift。
- **LLM Kernel 工作流：** 包含模型级服务命令、基准测试扫描、Kernel API 日志，
  以及 200 多个带有独立启动脚本的扁平任务目录，供后续优化循环使用。
- **开放前沿：** FA4/MHA 和类似 GEMM 的计算受限 Kernel 仍然更难优化；本仓库将
  失败和部分成功的尝试保持可见，让下一轮优化可以从证据出发，而不是依赖经验传言。

## 致谢

KDA-Pilot 中的“KDA”代表 **Kernel Design Agents**，这也是这些优化项目所基于的
Agent 方法论。感谢 MIT Han Lab 的原始项目：
[mit-han-lab/kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents)。
这里的 SGLang 服务 Kernel 任务由一个 fork
[BBuf/kernel-design-agents](https://github.com/BBuf/kernel-design-agents) 驱动
（接入每个任务的 `prompt.md`）；仓库附带的
[`external/KernelWiki`](external/KernelWiki) 知识库同样注明了对原始 KDA 项目的
致谢。

## Star 历史

[![Star History Chart](https://api.star-history.com/svg?repos=BBuf/KDA-Pilot&type=Date)](https://star-history.com/#BBuf/KDA-Pilot&Date)
