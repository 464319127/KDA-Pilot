请你基于现有 KDA kernel 任务提示，为 `b200_diffusion_fuse_scale_shift__multi_shape` 生成 Codex Goal 版 `plan.md`。

任务名：`b200_diffusion_fuse_scale_shift__multi_shape`
源任务 prompt：`/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/kernels/b200_diffusion_fuse_scale_shift__multi_shape/prompt.md`
最终输出：`/Users/bbuf/工作目录/Common/KDA-Pilot/codex-goal-diffusion/b200_diffusion_fuse_scale_shift__multi_shape/plan.md`

本轮只生成计划文件，不要实现 kernel，不要修改 `baseline/`、`solution/`、`bench/` 或 `docs/` 里的执行产物，不要运行 benchmark，不要进入远端机器，不要收集 profiler 或 NCU。源 prompt 里的远端、benchmark、NCU、优化循环要求应被转写为后续 Goal 的计划和验收条件，而不是本轮实际执行。

允许并要求读取：

- 上面的源任务 `prompt.md`，把它作为任务真源。
- `codex-goal-diffusion/README.md` 和 launcher 脚本约定，尤其是后续 Goal 的 cwd、`my/plan.md`、`baseline/`、`solution/`、`bench/`、`docs/` 产物布局。
- 源 prompt 明确要求遵循的 docs 或接口说明，用来准确抽取 baseline、ABI、workload、correctness、benchmark 和完成条件。
- OpenAI 官方 Codex Goal cookbook：`https://github.com/openai/openai-cookbook/blob/main/examples/codex/using_goals_in_codex.ipynb`。

不要编造尚未测得的性能数字；如果源 prompt 没有绝对性能阈值，计划应以“正确性通过 + 相对同环境 baseline 提升 + 证据充分”为完成标准。

请先使用已安装的 `brainstorming` skill 做设计梳理，但不要为了问问题而问问题：先探索上述上下文；如果源 prompt、README 和 docs 已经足够定义范围，就只提出必要的 2-3 个方案、给出推荐方案，并请求设计确认。只有存在会改变计划边界的真实歧义时才向我提一个简短阻塞问题。

设计确认后，使用 humanize 的 `gen-plan` 生成最终 `plan.md`。如该流程需要 draft 文件，可使用临时 draft；最终必须把完整计划保存到上面的输出路径。

最终 `plan.md` 必须是 Codex Goal 可执行契约，至少包含：

- Goal 最终状态：后续 `/goal follow the instruction in my/plan.md` 成功时应留下哪些文件、结果和证据。
- 源任务摘要：target GPU、目标 entry points/API、operation semantics、shape/workload coverage、baseline 类型和候选实现边界。
- Acceptance Criteria，使用 `AC-` 编号，覆盖 baseline 恢复、本地 ABI、workload/oracle、correctness、benchmark、prior-art、profiling evidence、结果报告和停止条件。
- Path Boundaries：允许读/复制/改写的路径、禁止污染的路径、任务目录内产物边界，以及是否允许上游集成。
- Baseline 恢复策略：如何从源 prompt 指定的 SGLang/依赖路径恢复 baseline，如何记录 upstream URL、commit、文件、版本和 ABI。
- Correctness 计划：reference/oracle、dtype 和容差策略、NaN/Inf 检查、positive/negative tests、边界 shape。
- Benchmark 计划：workload 来源、warmup/repeat、baseline 固化、候选对比、统计指标、日志位置和不能变更的方法学。
- GPU/容器/远端约束：根据源 prompt 写成后续 Goal 的执行约束；本轮不执行。
- KernelWiki 使用计划：明确后续 Goal 如何读取 `external/KernelWiki/SKILL.md` 或本地 `KernelWiki` skill 做 prior-art/设计检索，并记录采纳/拒绝理由。
- ncu-report-skill 使用计划：明确后续 Goal 在何时读取 `external/ncu-report-skill/SKILL.md` 或本地 skill，如何生成瓶颈证据，如何把 NCU 结果绑定到下一轮优化决策。
- 迭代策略：baseline -> first correct candidate -> benchmark -> profile -> bounded optimization loop -> promote/reject，含 search DAG/solutions 记录要求。
- 问题清单、默认假设、blocker、no-go 和停止条件。
