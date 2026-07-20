# KDA-Pilot：扩散模型 Kernel 任务

[English](README.md) | 简体中文

这是 KDA-Pilot 的 `diffusion/` 子目录，一个用于独立 GPU Kernel 优化任务的轻量级
提示词仓库。扩散模型任务面向 SGLang 扩散算子，但优化和基准测试工作区在运行时
不会修改、导入或 monkey-patch SGLang。

> 共享的 `external/` 知识子模块（`KernelWiki`、`ncu-report-skill`）位于仓库根目录，
> 与并行的 `llm/` 子目录并列。以下路径均相对于 `diffusion/`；从仓库根目录运行
> 启动脚本时，请使用 Launch 章节中带有 `diffusion/` 前缀的路径。

对于每个扩散模型任务，Agent 必须从最新的上游 SGLang `main` 提交中复制相关
Kernel 源码到任务的 `baseline/` 目录，通过与候选实现相同的低开销 ABI 暴露该复制的
基线，然后在任务目录内并排测试基线和候选实现。

## 目录结构

```text
docs/
  diffusion_correctness_contract.md
  diffusion_kernel_rules.md
  ghostty_claude_code_workflow.md
  diffusion_benchmark_shape_coverage.md
  standalone_diffusion_benchmark.md
  standalone_diffusion_benchmark_template.py
kernels/
  {b200,h200}_diffusion_qknorm_rope__multi_shape/
  {b200,h200}_diffusion_norm_infer__multi_shape/
  {b200,h200}_diffusion_group_norm_silu__multi_shape/
  {b200,h200}_diffusion_rotary_embedding__multi_shape/
  {b200,h200}_diffusion_fuse_scale_shift__multi_shape/
  {b200,h200}_diffusion_cutedsl_norm_tanh_mul_add__multi_shape/
  {b200,h200}_diffusion_cutedsl_norm_scale_shift__multi_shape/
  b200_diffusion_causal_conv3d_cat_pad__multi_shape/
  b200_diffusion_attention_concat_copy__multi_model/
  b200_diffusion_residual_gate_add__multi_shape/
  b200_ltx2_dual_modulate__bitwise/
  b200_ltx2_rms_adaln__bitwise/
  b200_ltx2_qknorm_split_rope__bitwise/
  b200_wan_vae_rmsnorm_silu__bitwise/
  b200_ernie_adaln_residual_gate__bitwise/
scripts/
  launch_kda_kernel_task.sh
  launch_kernels/
```

旧版的 SGLang overlay/export/capture 机制已被有意移除。仓库中没有
`kda_kernels/`，也不会向 SGLang checkout 应用补丁，更没有运行时安装路径。扩散模型
任务唯一依赖的 SGLang 内容，是复制到 `baseline/` 作为本地基准测试输入的上游源码。

## 扩散模型目录约定

每个扩散模型任务从以下干净结构开始：

```text
prompt.md       # Agent 任务卡
config.toml     # 任务的基准测试/构建默认配置
baseline/       # 由 Agent 生成的上游基线复制源码
solution/       # 优化后的候选实现
bench/          # 由 Agent 生成的独立基准测试/正确性测试框架
docs/           # 来源说明、基准测试日志、性能分析说明
```

核心基准规则位于
[`docs/standalone_diffusion_benchmark.zh-CN.md`](docs/standalone_diffusion_benchmark.zh-CN.md)。
扩散模型优化的约束位于
[`docs/diffusion_kernel_rules.zh-CN.md`](docs/diffusion_kernel_rules.zh-CN.md)。
标准回归网格位于
[`docs/diffusion_correctness_contract.md`](docs/diffusion_correctness_contract.md)。
生产 preset 和 shape 审计位于
[`docs/diffusion_benchmark_shape_coverage.md`](docs/diffusion_benchmark_shape_coverage.md)。
每个任务的 prompt 都要求 Agent 遵循这些文档。

## 基准测试原则

基线和候选必须通过匹配的本地接口进行比较。推荐使用本地直接 CUDA ABI：

- `language = "cuda"`
- `entry_point = "kernel.cu::<exported_symbol>"`
- `destination_passing_style = true`
- 直接导出 `TVM_FFI_DLL_EXPORT_TYPED_FUNC`
- 将输出 Tensor 作为末尾参数传入
- 使用 `at::cuda::getCurrentCUDAStream()` 启动 CUDA Kernel

基准测试必须使用固定工作负载行、每个工作负载隔离执行、预分配输出、预热、带内循环
放大的 CUDA Event 计时、交错 A/B 采样、严格的正确性检查和完整的来源记录。
每个扩散模型任务的 `bench/benchmark.py` 都应以
[`docs/standalone_diffusion_benchmark_template.py`](docs/standalone_diffusion_benchmark_template.py)
为起点。

## 启动

现有启动脚本仍会创建任务专属 worktree，并为 Agent 运行准备任务 prompt：

```bash
# 从仓库根目录运行
diffusion/scripts/launch_kernels/k03_b200_diffusion_qknorm_rope__multi_shape.sh
diffusion/scripts/launch_kernels/k20_b200_ltx2_dual_modulate__bitwise.sh
diffusion/scripts/launch_kernels/k21_b200_ltx2_rms_adaln__bitwise.sh
diffusion/scripts/launch_kernels/k22_b200_ltx2_qknorm_split_rope__bitwise.sh
diffusion/scripts/launch_kernels/k23_b200_wan_vae_rmsnorm_silu__bitwise.sh
diffusion/scripts/launch_kernels/k23_b200_ernie_adaln_residual_gate__bitwise.sh
```

设置 `KDA_NO_CLAUDE=1` 可以只准备 worktree 而不启动 Claude。
启动脚本默认使用当前 checkout 分支作为 `KDA_BASE_BRANCH`，因此任务 worktree 会
继承你正在测试的分支。只有在明确希望从另一个已提交的 ref 启动时，才设置
`KDA_BASE_BRANCH=<ref>`。

启动脚本还默认设置 `IS_SANDBOX=1`，供可能进入 root 所有的 Docker 环境的
Claude/Codex 会话使用。

设置 `KDA_BASH_BIN=/opt/homebrew/bin/bash` 可以强制启动脚本及其派生的 Humanize
Hook 使用现代 Bash；macOS 自带的 `/bin/bash` 3.2 会被拒绝。

如需在 Ghostty 中手动使用多个并行 pane，请参考
[`docs/ghostty_claude_code_workflow.md`](docs/ghostty_claude_code_workflow.md)。

## 维护

修改启动脚本后，检查其语法：

```bash
bash -n diffusion/scripts/launch_kda_kernel_task.sh diffusion/scripts/launch_kernels/*.sh
```

外部知识子模块仍然是可选的辅助材料：

```bash
git submodule update --init --recursive
```
