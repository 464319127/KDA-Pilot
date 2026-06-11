请你基于现有 KDA kernel 任务提示，为 `b200_diffusion_cutedsl_norm_tanh_mul_add__multi_shape` 生成 Codex Goal 版 `plan.md`。

任务名：`b200_diffusion_cutedsl_norm_tanh_mul_add__multi_shape`
源任务 prompt：`/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/kernels/b200_diffusion_cutedsl_norm_tanh_mul_add__multi_shape/prompt.md`
最终输出：`/Users/bbuf/工作目录/Common/KDA-Pilot/codex-goal-diffusion/b200_diffusion_cutedsl_norm_tanh_mul_add__multi_shape/plan.md`

本轮只生成计划文件，不要实现 kernel，不要修改 `baseline/`、`solution/`、`bench/` 或 `docs/` 里的执行产物，不要运行 benchmark，不要进入远端机器，不要收集 profiler 或 NCU。这个计划必须是全新的 greenfield Codex Goal：源 prompt 只提供任务定义和约束，不能把任何既有 KDA 优化过程或结果当作输入事实。源 prompt 里的远端、benchmark、NCU、优化循环要求应被转写为后续 Goal 的计划和验收条件，而不是本轮实际执行。

允许并要求读取（只读任务定义和公共规则）：

- 上面的源任务 `prompt.md`，把它作为任务入口、目标 entry points 和任务边界的真源；公共 `diffusion/docs/` 文档按补充约束也是 plan 的规则真源。
- `codex-goal-diffusion/README.md` 和 launcher 脚本约定，尤其是后续 Goal 的 cwd、`baseline/`、`solution/`、`bench/`、`docs/` 产物布局。
- `codex-goal-diffusion/PLAN_GENERATION_SUPPLEMENT.md`，这是所有 Codex Goal plan 生成 prompt 的公共补充约束；它对公共 `diffusion/docs/` 的读取要求和 shape-dispatch 计划要求优先于本 prompt 中更宽泛或更弱的表述。
- 源 prompt 明确点名的共享 docs、rules、contracts，例如 `diffusion/docs/` 下的公共 benchmark、correctness、kernel rules 和 shape coverage 文档，用来抽取任务约束、workload、oracle、benchmark 方法学和完成条件。
- 源 prompt 明确点名的静态接口/spec 文件，但只能在它显然不是运行记录、benchmark 结果、优化日志或 candidate 产物时读取；否则把它列为后续 Goal 需要自行恢复或确认的输入。
- OpenAI 官方 Codex Goal cookbook：`https://github.com/openai/openai-cookbook/blob/main/examples/codex/using_goals_in_codex.ipynb`。

禁止读取、参考、总结或复用任何 KDA 优化过程和结果：

- 不要读取 `/Users/bbuf/工作目录/Common/KDA-Pilot/diffusion/kernels/**` 下除当前任务源 `prompt.md` 和明确静态接口/spec 文件以外的任何文件。
- 不要读取或使用任何已有 `baseline/`、`solution/`、`bench/`、task-local `docs/`、`src/`、`tests/`、`profile/`、`ncu/`、`benchmark.csv`、`solutions.jsonl`、logs、captured tensors、profiler traces、NCU reports、draft/result summaries。
- 不要把既有 KDA run 的性能数字、结论、candidate 设计、失败路径或成功路径写入新 Goal；后续 Goal 必须从空白任务目录自行恢复 baseline、建立 harness、运行 correctness/benchmark/profile，并生成自己的证据。

不要编造尚未测得的性能数字；如果源 prompt 没有绝对性能阈值，计划应以“正确性通过 + 相对同环境 baseline 提升 + 证据充分”为完成标准。

请先使用已安装的 `brainstorming` skill 做设计梳理，但不要为了问问题而问问题：先探索上述上下文；如果源 prompt、README 和 docs 已经足够定义范围，就只提出必要的 2-3 个方案、给出推荐方案，并请求设计确认。只有存在会改变计划边界的真实歧义时才向我提一个简短阻塞问题。

设计确认后，使用 humanize 的 `gen-plan` 生成最终 `plan.md`。如该流程需要 draft 文件，可使用临时 draft；最终必须把完整计划保存到上面的输出路径。

最终 `plan.md` 必须是 Codex Goal 可执行契约，至少包含：

- Goal 最终状态：后续 `/goal follow the instruction in plan.md` 成功时应留下哪些文件、结果和证据。
- 源任务摘要：target GPU、目标 entry points/API、operation semantics、shape/workload coverage、baseline 类型和候选实现边界。
- Acceptance Criteria，使用 `AC-` 编号，覆盖 baseline 恢复、本地 ABI、workload/oracle、correctness、benchmark、prior-art、profiling evidence、结果报告和停止条件。
- Path Boundaries：允许读/复制/改写的路径、禁止读取的 KDA 过程/结果路径、禁止污染的路径、任务目录内产物边界，以及是否允许上游集成。
- Baseline 恢复策略：如何从源 prompt 指定的 SGLang/依赖路径恢复 baseline，如何记录 upstream URL、commit、文件、版本和 ABI。
- Correctness 计划：reference/oracle、dtype 和容差策略、NaN/Inf 检查、positive/negative tests、边界 shape。
- Benchmark 计划：workload 来源、warmup/repeat、baseline 固化、候选对比、统计指标、日志位置和不能变更的方法学。
- GPU/容器/远端约束：根据源 prompt 写成后续 Goal 的执行约束；本轮不执行。
- KernelWiki 使用计划：明确后续 Goal 如何读取 `external/KernelWiki/SKILL.md` 或本地 `KernelWiki` skill 做 prior-art/设计检索，并记录采纳/拒绝理由。
- ncu-report-skill 使用计划：明确后续 Goal 在何时读取 `external/ncu-report-skill/SKILL.md` 或本地 skill，如何生成瓶颈证据，如何把 NCU 结果绑定到下一轮优化决策。
- 迭代策略：baseline -> first correct candidate -> benchmark -> profile -> bounded optimization loop -> promote/reject，含全新 search DAG/solutions 记录要求。
- 问题清单、默认假设、blocker、no-go 和停止条件。
