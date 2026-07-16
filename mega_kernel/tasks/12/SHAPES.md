# 真实 shape(bs=1 serving;开工先实测复核)

TP8,world=8,NVLS multimem(任务 01 的 workspace/协议直接复用)。

| AR 站点(157 次/迭代) | producer GEMM | AR payload | 后续 |
|---|---|---|---|
| attn 输出(78/iter) | o_proj RowParallel [T, 8头×256]×[2048,6144],bf16,每 rank 部分和 | [T,6144] bf16(T∈{1,6},12-74KB) | +residual add + rmsnorm(任务01已融合在 AR 里) |
| MoE 输出(75/iter) | MoE finalize 输出(defer 路径) | 同上 | 同上 |

字节账:o_proj 权重每 rank 2048×6144×2B=25MB @3.7TB/s ≈ 6.8µs;AR 独立 kernel 7.6µs
——**链条里通信几乎和 GEMM 一样贵**,而 payload 只有几十 KB,纯延迟。融合目标:
GEMM+AR 链 ≤ GEMM+3µs 等效。

先决检查(开工第一件事):确认 nvjet 不可插 epilogue(闭源)→ P1 必须自带 GEMM;
用任务 10 的结论共享 GEMM 组件;测 multimem st 从 GEMM epilogue 发出的合法性
(mc_ptr 映射在 graph capture 下的可见性,任务 01 solution 里有全部 workspace 细节)。

复核:profiler 拆 o_proj→AR 的实际间隙;dump 一次真实 o_proj 输入/权重 stride。
