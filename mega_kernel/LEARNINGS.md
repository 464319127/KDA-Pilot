# 第一轮战役结论(2026-07-09/10,旧任务 01/02,已删除;完整报告在 common/prior_art/)

新任务开工前必读——这些是花了真机时间买来的否定结论与协议,不要重蹈。

## 判死的方向(不要再做)

1. **fp8 小 M dense GEMM 打不过 cuBLAS bf16**(旧 01):CUTLASS SM100 blockwise 赢了
   DeepGEMM(1.128×)但对 cuBLAS bf16 全线 0.70×。M≤6 时 nvjet bf16 splitK 跑
   ~3.7TB/s,所有 fp8 路径(DeepGEMM 含)被 per-128-K-tile 变换串行 + CLC/TMEM 机制
   开销压在 1.6-2.1TB/s——fp8 字节优势在这个 grid 尺寸不可兑现。**bf16-dense 就是
   生产解**(已上线,SGLANG_BS1_BF16_DENSE=1)。
2. **自研 unicast-push allreduce**(旧 02):正确但 36.6µs,~25µs 是协议固定成本
   (world=2 与 8 同速)。想快只有 NVLS multimem(multicast st + 交换机内 ld_reduce)
   ——flashinfer mnnvl kernel 已是该路径(8.3µs),所以新任务 01 是**移植+特化它**,
   不是重造传输。

## 有效的协议(直接复用)

- **cold-L2 基准协议**:回放同权重测出来的是 L2(5-6TB/s)不是 DRAM;权重类 kernel
  轮转 ≥48 份拷贝。(activation 类 KB 级 payload 影响小,但写 harness 时声明口径。)
- **8 卡 AR harness 口径**(common/harness/ar_bench.py):单进程 8 卡 + peer access,
  每卡 50-round CUDA graph 并发 replay,wall/round;spin/flag 类 kernel 的 NCU 必须
  `--replay-mode application`(kernel-replay 会死锁);1000-replay 位级稳定性是
  flag/epoch 竞态的标准暴露手段;图捕获需显式 per-device stream。
- **PDL 陷阱**(旧 01 round5):把 split-K 归约 PDL 链到 producer GEMM 上是无序 RAW
  (CUTLASS 在 input-consumed 就触发)——虚假加速 1.186→诚实 1.128。PDL 只在真依赖
  边界安全。
- **serving e2e 是唯一裁判**:隔离加速 ≥1.3× 且 e2e sanity 不倒退才 promote;
  e2e 口径 = /scratch/glm52_blog_bench/benchmark_glm52_bs1.py(sanity 1×40,官方 3×40)。

## 当前基线(2026-07-10)

- **376.06 tok/s 官方 3×40**(sglang main 87992eeec + 117 行移植 diff,见
  /personal/glm52_backup_20260710/patches_main/main_port_full.diff),accept 3.865。
- 运行时气泡已被 main 消除(>1ms gap = 0);剩余全是 kernel 时间。每迭代预算:
  dense+moe GEMM 7.55ms(cubin,不碰)、moe_aux 1.82、AR 1.40、elementwise 1.13、
  attention 0.94、norm_rope 0.68、quant 0.16(sum 口径,含多流重叠;span ~10.3ms)。
- 到 400 需 −0.62ms/iter → 本轮三任务合计预期 −0.5~−0.9ms。
