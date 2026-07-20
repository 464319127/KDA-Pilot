# Ghostty + Claude Code 工作流

[English](ghostty_claude_code_workflow.md) | 简体中文

使用 Ghostty 手动运行并行的 Claude Code 优化会话。启动脚本会为每个任务创建一个
独立的 Git worktree，因此每个 pane 都可以处理不同的 Kernel 目录，不会共享生成的
文件、Humanize 状态或 Git 提交。

本工作流适用于当前的独立扩散模型任务布局。旧版 SGLang overlay、export 和 shape
capture 机制已被有意移除。现在每个扩散模型任务都会将上游 SGLang Kernel 源码复制到
自己的 `baseline/` 目录，并通过匹配的本地 ABI 将复制的基线与候选实现进行基准测试。

## 前置条件

从包含当前任务骨架的已提交分支开始：

```bash
cd /Users/bbuf/工作目录/Common/KDA-Pilot
git status
git log --oneline -3
```

检查启动脚本：

```bash
bash -n diffusion/scripts/launch_kda_kernel_task.sh diffusion/scripts/launch_kernels/*.sh
```

如果任务需要远程 GPU 验证，使用与目标主机匹配的 Claude/Codex 远程 Skill：

```bash
ls ~/.codex/skills/ion-b200/SKILL.md
ls ~/.codex/skills/ion8-h200/SKILL.md
ls ~/.codex/skills/ion9-h200/SKILL.md
```

B200 任务应在 B200 上运行基准测试和性能分析，H200 任务应在 H200 主机上运行。
远程运行前，检查主机别名是否可访问并且存在空闲 GPU：

```bash
ssh -o ConnectTimeout=5 ion-b200  'hostname && nvidia-smi --query-gpu=name --format=csv,noheader | head -1'
ssh -o ConnectTimeout=5 ion8-h200 'hostname && nvidia-smi --query-gpu=name --format=csv,noheader | head -1'
ssh -o ConnectTimeout=5 ion9-h200 'hostname && nvidia-smi --query-gpu=name --format=csv,noheader | head -1'
```

对于 FLUX 或 FLUX.2 等需要 gated 模型的 shape 审计重跑，在启动 shell 中导出已授权的
Hugging Face token：

```bash
export HF_TOKEN=hf_...
```

普通的独立 Kernel 基准测试不应在运行时导入或修改 SGLang。修改任务前先阅读以下文档：

- [`standalone_diffusion_benchmark.zh-CN.md`](standalone_diffusion_benchmark.zh-CN.md)
- [`diffusion_kernel_rules.zh-CN.md`](diffusion_kernel_rules.zh-CN.md)
- [`diffusion_correctness_contract.zh-CN.md`](diffusion_correctness_contract.zh-CN.md)
- [`diffusion_benchmark_shape_coverage.zh-CN.md`](diffusion_benchmark_shape_coverage.zh-CN.md)

## Ghostty 快捷键

| 快捷键 | 操作 |
|---|---|
| `Cmd + D` | 将当前 pane 向右分屏 |
| `Cmd + Shift + D` | 将当前 pane 向下分屏 |
| `Cmd + W` | 关闭当前 pane/标签页/窗口表面 |
| `Cmd + T` | 打开新标签页 |
| `Cmd + Shift + [` | 上一个标签页 |
| `Cmd + Shift + ]` | 下一个标签页 |
| `Cmd + ,` | 打开 Ghostty 配置 |
| `Cmd + Shift + ,` | 重新加载 Ghostty 配置 |
| `Ctrl + backquote` | 切换快速终端 |
| `Cmd + backquote` | 切换快速终端 |

使用以下命令验证配置：

```bash
ghostty +validate-config
```

## 并行策略

本文档不设优先级分工。选择任意一个尚未被占用且目标 GPU 可用的扩散模型任务。
使用四个 pane 时，可以采用以下实用模式：

1. 在一个 Ghostty 标签页中打开四个 pane。
2. 在每个 pane 中启动一个 `diffusion/scripts/launch_kernels/kXX_*.sh` 脚本。
3. 某个 pane 完成或因硬件不可用而阻塞时，启动另一个未占用的任务。
4. 将 B200 任务保持在 B200 上，将 H200 任务保持在 H200 上。

`K` 编号只是 `diffusion/scripts/launch_kernels/` 下的启动文件编号，不代表优化优先级。

## 启动命令

从仓库根目录运行脚本：

```bash
cd /Users/bbuf/工作目录/Common/KDA-Pilot
./diffusion/scripts/launch_kernels/k03_b200_diffusion_qknorm_rope__multi_shape.sh
```

对于临时任务目录：

```bash
./diffusion/scripts/launch_kda_kernel_task.sh diffusion/kernels/<task-folder>
```

设置 `KDA_NO_CLAUDE=1` 可以创建任务 worktree 并打印命令，但不启动 Claude：

```bash
KDA_NO_CLAUDE=1 ./diffusion/scripts/launch_kernels/k03_b200_diffusion_qknorm_rope__multi_shape.sh
```

启动脚本会创建：

- 一个任务专属的 Git 分支；
- 一个任务专属的 Git worktree；
- 一个名为 `kda-base/<task-label>-<run-id>` 的固定本地 review-base 分支；
- 根据任务 `prompt.md` 生成的 `.humanize/kernel-agent/draft.md`。

使用启动脚本打印出的 review-base 分支进行 RLCR。不要自行猜测分支名。

有用的启动脚本覆盖项：

```bash
KDA_BASE_BRANCH=<ref>   # 可选；默认使用当前 checkout 分支
KDA_WORKTREE_BASE=/path/to/KDA-Pilot-worktrees
KDA_RUN_ID=my-run
KDA_BRANCH_PREFIX=kda
KDA_NO_CLAUDE=1
CLAUDE_BIN=claude
CLAUDE_MODEL=opus
CLAUDE_EFFORT=max
KDA_BASH_BIN=/opt/homebrew/bin/bash                 # 启动脚本和 Hook 使用的现代 bash；
                                                   # /bin/bash 3.2 会被拒绝
IS_SANDBOX=1                                      # 默认值；传入 Claude 环境
                                                  # 供 root 所有的 Docker 会话使用。
HUMANIZE_CODEX_BYPASS_SANDBOX=true                # 默认值；传入 Claude 环境，
                                                  # 让 RLCR 循环中的 Codex 跳过
                                                  # 每次调用的沙箱/审批提示。
                                                  # 设置为除 true|1 外的任何值即可
                                                  # 重新启用沙箱。
```

## Humanize 流程

启动脚本会进入生成 worktree 中的任务目录。从那里生成计划：

```text
/humanize:gen-plan --input .humanize/kernel-agent/draft.md --output .humanize/kernel-agent/refined-plan.md --direct
```

默认不要运行 `refine-plan`。只有当生成的计划包含 `CMT:` / `ENDCMT`、
`<cmt>` / `</cmt>` 或 `<comment>` / `</comment>` 等手动审查注释块时，才使用它。

使用启动脚本打印出的 review-base 分支启动 RLCR：

```text
/humanize:start-rlcr-loop diffusion/kernels/<kernel-folder>/.humanize/kernel-agent/refined-plan.md --skip-quiz --claude-answer-codex --max 12 --codex-model gpt-5.5:xhigh --codex-timeout 5400 --base-branch <printed-kda-base-branch>
```

RLCR 启动后，从生成的
`.humanize/rlcr/<timestamp>/round-0-prompt.md` 继续。

## 任务约定

每个扩散模型任务初始只包含：

```text
prompt.md
config.toml
baseline/.gitkeep
solution/.gitkeep
bench/.gitkeep
docs/.gitkeep
```

首个 Agent 里程碑必须填充：

```text
baseline/
  复制的上游 SGLang 源码文件
  本地基线 ABI 封装
solution/
  使用相同 ABI 的候选实现
bench/
  workloads.json
  correctness.py
  benchmark.py
  adapter.py
docs/
  baseline_source.md
  benchmark_method.md
  benchmark_preset_audit.md
  run_log.md
```

`baseline_source.md` 必须记录在恢复基线时解析到的最新上游 SGLang `main` 提交，
以及复制的文件和本地适配器修改。

如果基准测试在运行时导入 SGLang、修改 SGLang checkout、比较不同的封装开销、静默
跳过生产工作负载、在调优后修改工作负载却没有重新测量两侧，或保留没有来源信息的
计时数字，则该基准测试无效。

## 启动脚本

共有 16 个启动脚本：2 个参考任务和 14 个扩散模型任务。下面的扩散模型条目仅按
启动器编号排列。

| 启动脚本 | 目录 |
|---|---|
| `k01_b200_int8_scaled_mm.sh` | `diffusion/kernels/b200_int8_scaled_mm__m64_n2048_k2048_bias` |
| `k02_b200_fa4_mha.sh` | `diffusion/kernels/b200_fa4_mha__bf16_head128_total32768` |
| `k03_b200_diffusion_qknorm_rope__multi_shape.sh` | `diffusion/kernels/b200_diffusion_qknorm_rope__multi_shape` |
| `k04_h200_diffusion_qknorm_rope__multi_shape.sh` | `diffusion/kernels/h200_diffusion_qknorm_rope__multi_shape` |
| `k05_b200_diffusion_norm_infer__multi_shape.sh` | `diffusion/kernels/b200_diffusion_norm_infer__multi_shape` |
| `k06_h200_diffusion_norm_infer__multi_shape.sh` | `diffusion/kernels/h200_diffusion_norm_infer__multi_shape` |
| `k07_b200_diffusion_group_norm_silu__multi_shape.sh` | `diffusion/kernels/b200_diffusion_group_norm_silu__multi_shape` |
| `k08_h200_diffusion_group_norm_silu__multi_shape.sh` | `diffusion/kernels/h200_diffusion_group_norm_silu__multi_shape` |
| `k09_b200_diffusion_rotary_embedding__multi_shape.sh` | `diffusion/kernels/b200_diffusion_rotary_embedding__multi_shape` |
| `k10_h200_diffusion_rotary_embedding__multi_shape.sh` | `diffusion/kernels/h200_diffusion_rotary_embedding__multi_shape` |
| `k11_b200_diffusion_fuse_scale_shift__multi_shape.sh` | `diffusion/kernels/b200_diffusion_fuse_scale_shift__multi_shape` |
| `k12_h200_diffusion_fuse_scale_shift__multi_shape.sh` | `diffusion/kernels/h200_diffusion_fuse_scale_shift__multi_shape` |
| `k13_b200_diffusion_cutedsl_norm_tanh_mul_add__multi_shape.sh` | `diffusion/kernels/b200_diffusion_cutedsl_norm_tanh_mul_add__multi_shape` |
| `k14_h200_diffusion_cutedsl_norm_tanh_mul_add__multi_shape.sh` | `diffusion/kernels/h200_diffusion_cutedsl_norm_tanh_mul_add__multi_shape` |
| `k15_b200_diffusion_cutedsl_norm_scale_shift__multi_shape.sh` | `diffusion/kernels/b200_diffusion_cutedsl_norm_scale_shift__multi_shape` |
| `k16_h200_diffusion_cutedsl_norm_scale_shift__multi_shape.sh` | `diffusion/kernels/h200_diffusion_cutedsl_norm_scale_shift__multi_shape` |

## 扩散模型任务卡

这些任务卡是精简的选择器。完整要求位于各任务的 `prompt.md` 中。

| 启动器 | 系列 | 作为基线复制的 SGLang 入口 |
|---|---|---|
| K3 / K4 | QKNorm + RoPE | `qknorm_rope:fused_inplace_qknorm_rope` |
| K5 / K6 | Norm infer | `norm:norm_infer`、`rmsnorm_onepass:triton_one_pass_rms_norm` |
| K7 / K8 | GroupNorm + SiLU | `group_norm_silu:apply_group_norm_silu`、`triton.group_norm_silu:triton_group_norm_silu` |
| K9 / K10 | Rotary embedding | `rotary:apply_rotary_embedding`、`ltx2_rotary:apply_ltx2_split_rotary_emb` |
| K11 / K12 | Scale-shift | `scale_shift:fuse_scale_shift_kernel`、`fuse_layernorm_scale_shift_gate_select01_kernel`、`fuse_residual_layernorm_scale_shift_gate_select01_kernel` |
| K13 / K14 | Norm + tanh + mul + add | `norm_tanh_mul_add_norm_scale:fused_norm_tanh_mul_add`、`fused_norm_tanh_mul_add_norm_scale` |
| K15 / K16 | Norm-scale-shift | `scale_residual_norm_scale_shift:fused_norm_scale_shift`、`fused_scale_residual_norm_scale_shift` |

shape 和实时 preset 覆盖情况请参考
[`diffusion_benchmark_shape_coverage.zh-CN.md`](diffusion_benchmark_shape_coverage.zh-CN.md)。
如果覆盖文档表示某个 preset 被阻塞，不要自行编造 shape 行。应在 B200/H200 上重新
运行模型并记录有效的原生捕获，或者在任务内的 `docs/benchmark_preset_audit.md`
中添加实时 no-call 证明。

## 完成日志

每个启动的任务保留一行记录，同时记录成功和 no-go 结果。

| 任务 | 状态 | 结果 | 主机/GPU | 提交 | Claude Code 成本 |
|---|---|---|---|---|---|
| K3 | pending | 待填写 | 待填写 | 待填写 | 待填写 |
| K4 | pending | 待填写 | 待填写 | 待填写 | 待填写 |
| K5 | pending | 待填写 | 待填写 | 待填写 | 待填写 |
| K6 | pending | 待填写 | 待填写 | 待填写 | 待填写 |
| K7 | pending | 待填写 | 待填写 | 待填写 | 待填写 |
| K8 | pending | 待填写 | 待填写 | 待填写 | 待填写 |
| K9 | pending | 待填写 | 待填写 | 待填写 | 待填写 |
| K10 | pending | 待填写 | 待填写 | 待填写 | 待填写 |
| K11 | pending | 待填写 | 待填写 | 待填写 | 待填写 |
| K12 | pending | 待填写 | 待填写 | 待填写 | 待填写 |
| K13 | pending | 待填写 | 待填写 | 待填写 | 待填写 |
| K14 | pending | 待填写 | 待填写 | 待填写 | 待填写 |
| K15 | pending | 待填写 | 待填写 | 待填写 | 待填写 |
| K16 | pending | 待填写 | 待填写 | 待填写 | 待填写 |

已关闭任务的字段：

- outcome：`IMPROVEMENT`、`NO-GO` 或 `BLOCKED`；
- 测得的加速比或几何平均值；
- 主机、GPU ID 和 GPU 型号；
- 基线提交/源码和候选提交；
- 基准测试命令和结果路径；
- Claude Code `/usage` 成本。
