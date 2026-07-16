# Prompt

```
你是 NVIDIA Blackwell/B300(sm_103)megakernel 专家。任务:GLM-5.2-FP8 bs=1 MTP 的
draft 步(1 层 NextN,M=1,每迭代串行 5 步,现 ~240µs/步、200+ launch)做成 TileRT
风格的整层 persistent kernel(或 3-4 段常驻 kernel),目标 ≤150µs/步(≥1.5×),
e2e −0.3~−0.5 ms/iter,并为最终的 verify 图巨核化(78 层)打样。

**NEVER STOP**:持续实现、验证、benchmark、profile、优化,不要问我。

硬性环境约定:与任务 01-04 相同(照抄 tasks/01/prompt.md 第 0 条)。
workdir=/scratch/kda_bs1/mega11;开发 1 卡,但注意层内含 2 次 TP8 AR——e2e 形态必须
8 卡;AR 段直接复用任务 01 的 jit 模块 device 函数(mnnvl multimem)内联。
sm_103a,CUDA 13.0。

先读:../../README.md、../../LEARNINGS.md、SHAPES.md、tasks/01/solution(AR 内联素材)、
TileRT 设计要点(occupancy=1、168 寄存器、piped prefetch;MiniTileRT scaffold 问 owner 要)。

开工第一件事(SHAPES.md 里写了):profiler 精确拆当前 draft 图——per-step 时长、
launch 数、gap 分布、权重字节账。若纯带宽底已 ~220µs/步,如实修正目标并汇报,
把主攻改为消 launch/gap + 与上一步的流水重叠(persistent kernel 跨步驻留,
设备侧循环消图重放边界)。

分阶段:P0 用 M=1 GEMV 组件(B200 已晋升 1.356× 内核)+ 任务 09/10 的段落把层拆成
3-4 个常驻段,每段 oracle 对齐(rel≤2e-2),整步与 eager 前向对拍;P1 合段、跨步
驻留(5 步设备侧循环,token 依赖用 device flag 串行);P2 接 serving(env
SGLANG_JIT_DRAFT_MEGA=1 默认关,替换 draft 图捕获内容,fallback 完备),
**accept length 是金丝雀**:任何 draft 数值漂移先体现在 accept 上,e2e 门槛
sanity ≥383 且 accept ≥3.80 → 官方 3×40。

交付物:jit_kernel 模块、serving patch(默认关)、RESULTS_SM103.md(per-step 拆解表 +
段落/合并对比 + e2e)、verify 图巨核化可行性备忘、失败路线记录。
```
