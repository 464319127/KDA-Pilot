# Prompt

```
你是 NVIDIA Blackwell/B300(sm_103)通信×计算融合专家。任务:GLM-5.2-FP8 bs=1 的
GEMM+AllReduce 融合 kernel——RowParallel producer GEMM(o_proj,[T,*]×[2048,6144],
T∈{1,6})的 epilogue 直接 NVLS multimem 存储 + 到达 flag,消费端 prologue 做
ld_reduce+add+rmsnorm,消灭独立 AR kernel(任务 01 已把它压到 7.6µs 的协议下界,
再往下只有把 kernel 边界本身抹掉)。链条目标 ≥1.25×(GEMM 段不掉出 nvjet 95%),
e2e −0.3~−0.6 ms/iter(两个 AR 站点合计 157 次/迭代)。

**NEVER STOP**:持续实现、验证、benchmark、profile、优化,不要问我。

硬性环境约定:与任务 01-04 相同(照抄 tasks/01/prompt.md 第 0 条:rx devbox 8×B300
申请、proxychains、权重 NVMe、serving 基线 381.42 重建)。workdir=/scratch/kda_bs1/mega12;
本任务是集合通信,开发即需 8 卡(与 serving 错峰)。sm_103a,CUDA 13.0。

先读:../../README.md、../../LEARNINGS.md(单播路线判死:25µs 协议成本;multimem 是
唯一快路)、tasks/01/solution(拿来即用的 multimem workspace/epoch/flag 全套 device
代码 + 8 卡 harness common/harness/ar_bench.py)、SHAPES.md(字节账:AR 与 GEMM 等贵,
纯延迟属性)。

分阶段:
P0 把任务 01 的 AR 拆成可从任意 kernel 调用的 device 函数库(multimem st / 到达 flag /
epoch 轮转 / ld_reduce),独立 harness 重现 7.6µs 与位级一致——这一步纯重构,必须位级。
P1 GEMM 侧:从 B200 已晋升 M≤6 bf16 内核或 CUTLASS epilogue-visitor 起步(禁止从零
写 GEMM),epilogue 每算完一个输出 tile 就 multimem st + flag;消费端(下一层第一个
kernel 或专用小尾块)ld_reduce+add+rmsnorm。正确性:fp32 oracle rel≤2e-2(归约序
必然变),1000-replay 稳定,graph 可捕获。
P2 serving 接入:先 o_proj→AR 站点(SGLANG_JIT_GEMM_AR=1 默认关,fallback 完备),
sanity ≥383 + accept ≥3.80 → 官方 3×40;MoE 输出站点与任务 09 协同评估。

风险与如实汇报点:GEMM 自持(不用 nvjet)可能在 GEMM 段掉带宽——若 P1 的 GEMM 段
达不到 nvjet 90%,把结论(差距、原因、NCU)写清并停在 P0 交付(device 函数库本身
就是任务 09/11 巨核的 AR 内联素材,独立有价值)。

交付物:device 函数库 + 融合 kernel + serving patch(默认关)+ RESULTS_SM103.md
(链条对比 + GEMM 段 BW 表 + e2e)+ 失败路线记录。
```
