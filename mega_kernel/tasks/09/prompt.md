# Prompt

```
你是 NVIDIA Blackwell/B300(sm_103)MoE kernel 专家。任务:为 GLM-5.2-FP8 bs=1 写
per-layer 的 MoE persistent 巨核——routing→gemm1→SiLU·mul→gemm2→top8 加权 finalize
(+shared add)一个 kernel 完成,替换 flashinfer trtllm-gen cubin 链(5-7 个 launch,
T=6 实测 ~55-60µs/层,75 层/迭代)。目标 ≤42µs/层(≥1.35×),e2e −0.6~−1.2 ms/iter。
这是任务板上最大的池子,也是最难的任务;正确性口径 = fp32 oracle rel≤2e-2 + e2e
质量门(非位级,归约结构必然不同)。

**NEVER STOP**:持续实现、验证、benchmark、profile、优化,不要问我。

硬性环境约定(与任务 01-04 相同,详见 tasks/01/prompt.md 第 0 条并照抄执行):
rx devbox 池申请 8×B300(rxp devbox acquire --gpu B300 --count 8 --image
lmsysorg/sglang:latest --name glm52-bs1-opt),proxychains 标准包装,权重
/cluster-storage → /scratch/models NVMe,serving 基线 = sglang main 87992eeec +
/personal/glm52_backup_20260710/patches_main/main_port_full.diff + 任务01接入器
(KDA-Pilot mega_kernel/tasks/01/solution/serving/sglang_patcher.py apply),
基线吞吐 381.42(env: BF16_DENSE + 2×DEFER + JIT_MNNVL_AR×2)。
workdir=/scratch/kda_bs1/mega09;kernel 开发 1 卡,e2e 8 卡。sm_103a,CUDA 13.0。

先读:../../README.md(kernel 占比与关键路径教训)、../../LEARNINGS.md(判死方向:
fp8 dense 小 M 打不过 cuBLAS 是 DENSE 的结论,MoE fp8 无 bf16 对手,不适用)、
SHAPES.md(链条与字节账:top-8 实际读 ~38MB/层,当前 39µs 离带宽有 3× 余量)。

设计要点(参考,不设限):
- persistent 单 kernel:CTA 常驻,内部阶段化(routing 由 1 CTA 算完广播 → 各 CTA 认领
  (expert,tile) 工作项 → gemm1+act 写 smem/寄存器 → gemm2 → 原子/树归约 finalize)。
- fp8 128×128 block 反量化在 tile 内做;权重流用 TMA/LDG.128;T=6 的激活常驻寄存器。
- 参考 TileRT 形态:occupancy=1、高寄存器、piped prefetch(讨论材料在仓库根
  TileRT_讨论材料.md 的知识沉淀里,MiniTileRT scaffold 可要来参考)。
- CUTLASS SM100 blockwise GEMM 可作为 gemm 内核起点(LEARNINGS:它对 DeepGEMM 1.128×)。
- 分阶段落地:P0 先把链条中"routing+topk+pack"和"act+gemm2 前半"各自融成段(2-3 launch),
  每段 oracle 对齐;P1 再合成单 kernel。每步都要能跑 e2e 质量门,防止一步到位翻车。
- CUDA graph 可捕获;spin/barrier 用 application-replay NCU。

门槛:隔离(cold-L2,48 份权重轮转)T=6 ≤42µs/层 且 T=1 不劣化;fp32 oracle rel≤2e-2;
接入 serving(env SGLANG_JIT_MOE_MEGA=1 默认关,moe_runner 分发,非覆盖形态 fallback)
后 sanity ≥383 且 accept ≥3.80 → 官方 3×40 记录。

交付物:jit_kernel 模块、serving 接入 patch(默认关)、RESULTS_SM103.md(分段与单核
的对比表 + NCU 证据 + e2e)、失败路线记录。
```
