# Prompt

```
你是 NVIDIA Blackwell/B300(sm_103)GEMM+epilogue 融合专家。任务:GLM-5.2-FP8 bs=1
的 MLA a-path 竖向融合——fused_qkv_a 的 bf16 小 M GEMM([T,6144]×[6144,2624],T∈{1,6})
尾部直接完成 双RMSNorm→RoPE→fp8量化→KV写入,5-6 个 kernel 变 1 个。链条 ~17-19µs →
≤13µs(≥1.3×),78+ 次/迭代在串行流上,e2e −0.25~−0.45 ms/iter。

**NEVER STOP**:持续实现、验证、benchmark、profile、优化,不要问我。

硬性环境约定:与任务 01-04 相同(照抄 tasks/01/prompt.md 第 0 条:rx devbox 申请、
proxychains、权重 NVMe、serving 基线 381.42 重建流程)。workdir=/scratch/kda_bs1/mega10;
kernel 开发 1 卡。sm_103a,CUDA 13.0。

先读:../../README.md、../../LEARNINGS.md、SHAPES.md。核心约束:**GEMM 本体已在带宽
顶(nvjet 3.7TB/s),不许倒退超 5%**——赢点全在把 4-5 个 2-3µs 的尾巴 kernel 吸进
GEMM 收尾,以及消掉中间张量的 global 往返。B200 战役有已晋升的 M≤6 GEMV/GEMM 内核
(1.356×,问 owner 要 KDA 旧战役 solution)作起点,禁止从零写 GEMM。

分阶段:P0 复刻裸 GEMM 达 nvjet ≥95%(cold-L2 48 份权重)——达不到就先攻这个,达不到
90% 直接汇报停止;P1 逐段挂 epilogue(每挂一段与 standalone kernel 位级对拍该段输出);
P2 接 serving(env SGLANG_JIT_MLA_APATH=1 默认关,MLA forward 分发,fallback 完备),
sanity ≥383 + accept ≥3.80 → 官方 3×40。

交付物:jit_kernel 模块、serving patch(默认关)、RESULTS_SM103.md(每段前后对比 +
BW 表 + e2e)、失败路线记录。
```
