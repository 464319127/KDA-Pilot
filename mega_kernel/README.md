# mega_kernel — GLM-5.2-FP8 bs=1 解码 kernel 战役(第二轮:flashinfer 开源 kernel → sglang jit_kernel)

目标:8×B300(sm_103)上 GLM-5.2-FP8 bs=1 MTP(EAGLE 5-1-6)解码吞吐
**≥400 tok/s**(官方 3×40 口径)。任务 01 已晋升:376.06 → **381.42**(mnnvl AR jit 特化,+1.4%),当前还差 ~0.47 ms/迭代。

策略:serving 热路径里凡是**有源码**的 kernel(flashinfer JIT 源 + sglang 自家),
移植进 sglang `python/sglang/jit_kernel/` 后针对 bs=1 真实 shape 特化。cubin 部分
(cuBLAS nvjet、trtllm-gen GEMM/FMHA)不碰。**先移植后优化:移植版必须逐位一致、
性能持平、e2e 吞吐不变(env 门控默认关),然后才允许改。**

## 任务索引(替换第一轮的 01/02;第一轮结论见 LEARNINGS.md,必读)

| 任务 | 对象(全部 flashinfer JIT 源码) | 池子 ms/iter | 目标 | 一键启动 |
|---|---|---:|---|---|
| [01](tasks/01/) ✅晋升 | mnnvl 融合 AR+add+rmsnorm(`trtllm_mnnvl_allreduce.cuh`) | 1.31 | **已交付:8.3→7.6µs,官方 381.42(+1.4%)**;7µs 判定为 NVLS 延迟下界(位级约束内) | 完成,RESULTS 见 tasks/01/ |
| [02](tasks/02/) | RMSNorm(`norm.cu`)+ RopeQuantize(`rope.cu`)横/纵向融合包 | 0.60 | 融合 ≥1.4×,e2e −0.2~0.3ms | `scripts/launch_kernels/k02_b300_norm_rope_pack_jit_bs1.sh` |
| [03](tasks/03/) | MoE sigmoid top-8/256 routing(`RoutingKernel.cuh`)+ topk_small_batch | 0.64 | routing ≤3.5µs,e2e −0.15~0.25ms | `scripts/launch_kernels/k03_b300_moe_routing_topk_jit_bs1.sh` |

合计预期 −0.5~−0.9 ms/iter → **400-410 tok/s**。

## 铁律

0. **只做增量修改,禁止从零重写**:P1 的每个优化都从 P0 移植的 flashinfer 源码出发
   (删死代码/常量化/调 launch 形态/融合相邻 kernel),每步保持位级一致再进下一步。
   第一轮教训:从零手写(mma GEMM、unicast AR)分别比现有实现慢 3×/4×。
1. **shape 全部来自 bs=1 真实 serving**(T∈{1,6},配置冻结的 EAGLE 5-1-6),每个任务
   开工先在箱上实测复核 SHAPES.md,不许用臆造 shape 调优。
2. **P0 移植阶段位级一致 + e2e 不倒退是硬门槛**,不过 P0 不许进优化。
3. serving 接入一律 env 门控、默认关;当前基线 **381.42** = 376.06 配置 + SGLANG_JIT_MNNVL_AR=1 SGLANG_JIT_MNNVL_AR_OPT=1。
4. promote = 隔离达标 + sanity ≥383 且 greedy 输出逐字一致 + 官方 3×40 记录(基线 381.42)。
5. 机器断开/过期不是终止(重连/重申请流程写在各 prompt 第 0 条)。

## 环境

- 机器:rx devbox `glm52-bs1-opt`(verda-b300-fin-03-1,8×B300 SXM6,x86)。
- serving 基线:sglang main `87992eeec` + `/personal/glm52_backup_20260710/patches_main/
  main_port_full.diff`(GLM 装载修复 + bf16-dense + fp8-defer),权重 NVMe
  `/scratch/models/glm52-fp8`,重启 ~13 分钟。
- 评测:`/scratch/glm52_blog_bench/benchmark_glm52_bs1.py`(sanity `--runs 1`,官方
  `--runs 3`);profile 用 /start_profile num_steps≤100(严禁不限步,曾把服务器
  OOM 打挂)。
- 公共资产:`common/harness/`(8 卡 AR harness)、`common/prior_art/`(第一轮完整
  RESULTS)、`LEARNINGS.md`(判死方向与协议)。

## 启动方式

```bash
# 例:启动任务 01(自动建 git worktree + Claude Code + Humanize RLCR 循环)
mega_kernel/scripts/launch_kernels/k01_b300_mnnvl_ar_jit_bs1.sh
```
三个任务相互独立可并行,但共享 devbox——kernel 开发各用 1 卡即可(任务 01 需 8 卡
时段性占用),e2e 验证需独占重启 serving,错峰进行。
