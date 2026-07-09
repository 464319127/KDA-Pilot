# Prompt

```
你是 NVIDIA Blackwell/B300(sm_103)NVLink 通信 kernel 专家。任务:把 GLM-5.2 FP8 bs=1
serving 里每迭代 160 次的融合 allreduce(+residual add+rmsnorm)从 8.7µs 压到 ≤5µs
(payload 仅 6×6144 bf16 = 73.7KB,单机 8×B300 全互联 NV18),并接回 serving。

**NEVER STOP**:持续实现、验证、benchmark、profile、优化,不要问我。

硬性环境约定:
1) 所有 GPU 命令在 rx devbox `glm52-bs1-opt` 上跑(`ssh glm52-bs1-opt`;devbox 过期的
   重建方法见 ../01_dense_fp8_gemm_bs1/prompt.md 第 1 条)。
2) 需要全部 8 卡(单进程多卡 + cudaDeviceEnablePeerAccess);workdir =
   /scratch/kda_bs1/meta03;与常驻 serving 进程可共存。
3) 编 sm_103a,CUDA 13.0。

现状(先读 RESULTS_SM103.md):
- v1 单播 push 原型(ar_oneshot.cu + ar_bench.py)正确性 8/8,但 36.6µs;探针证明
  ~25µs 是协议固定成本(world=2 与 world=8 几乎同速)→ 不要继续优化单播路线。
- 参考基线 flashinfer mnnvl `oneshotAllreduceFusionKernel` = 8.7µs(serving 图内实测),
  它快在 NVLS multimem(multicast store + 交换机内 ld_reduce)。

任务路线(按此顺序):
1) 路线 A(优先):在独立 harness 里直接驱动 flashinfer 的 mnnvl AR API
   (flashinfer.comm,workspace 初始化参考 sglang flashinfer_comm_fusion.py),
   复现 8.7µs,然后 NCU/时间戳拆解它的组成(multicast 写、栅栏、norm 段),
   攻残余开销:launch 维度、grid 规模、fence 粒度、norm 并入方式。理论下界 ~2-3µs。
2) 路线 B(若 A 撞墙):cuMulticastCreate/BindMem 自建 multimem 路径,单 kernel
   multimem.st + multimem.ld_reduce + add + rmsnorm,epoch 自轮转保证图可捕获
   (v1 的 epoch 机制可复用)。
3) 严禁的坑:同权重/同缓冲回放不影响本任务(通信量恒定),但 spin 类 kernel 的 NCU
   必须 --replay-mode application,否则死锁。

基准与门槛:
1) 计时口径:每卡一张 50-round CUDA graph,8 卡并发 replay,wall/round(v1 harness
   已实现该口径,直接复用)。
2) 正确性:fp32 oracle rel < 2e-2(out 与 residual_out 都要),1000 次 replay 无 flaky
   (flag/epoch 竞态的典型表现);跨 replay 逐位稳定。
3) promote:T=6 payload ≤5µs 且 T=1(draft 12KB payload)不劣化;NCU 证据。
4) 接回 serving:communicator.py / flashinfer_comm_fusion.py 增加 backend 选项,
   e2e 验证同 01 任务(sanity → 3×40 官方口径)。预期 e2e 收益 ~-0.5~-0.8ms/iter。

交付物:solution/ kernel 或 flashinfer 调参 patch、RESULTS_SM103.md 更新(组成拆解表
+ 前后对比)、serving e2e 数据、失败路线记录。
```
