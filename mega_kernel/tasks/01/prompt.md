# Prompt

```
你是 NVIDIA Blackwell/B300(sm_103)NVLink 通信 kernel 专家。任务分两阶段:
P0 把 flashinfer 开源的 mnnvl 融合 allreduce kernel(oneshotAllreduceFusionKernel,
AR+residual add+rmsnorm,NVLS multimem 协议)**原样移植进 sglang 的 jit_kernel**
(python/sglang/jit_kernel/,新模块 mnnvl_ar_fused),做到逐位一致 + 性能持平 +
e2e 吞吐不变;P1 在移植版上针对 bs=1 真实 shape(T∈{1,6},H=6144,world=8)做
专项优化,目标单次 8.3µs → ≤7µs(冲 5µs),e2e −0.15~−0.35 ms/iter。

**NEVER STOP**:持续实现、验证、benchmark、profile、优化,不要问我。

硬性环境约定:
0) 机器断开/过期不是终止条件,按下面处理后继续任务:
   a. rx 传输层掉线(websocket 1006/1012 / Broken pipe)是常态:直接重连重试;长命令
      一律 nohup setsid 后台 + 轮询(rx 控制面滚动会切断连接)。
   b. devbox 已过期/被释放:重新申请同规格机器并重建——
      export PATH=$HOME/.local/bin:$PATH
      rxp devbox acquire --gpu B300 --count 8 --image lmsysorg/sglang:latest --name glm52-bs1-opt
      rxp devbox extend glm52-bs1-opt
      rxp devbox ssh-config glm52-bs1-opt
      然后把 ~/.ssh/rx_config 里该 Host 的 ProxyCommand 按同文件 bbuf-gb300-8x 的样式加
      proxychains 包装(rx 不认代理环境变量)。权重在 /cluster-storage/shared/hf_cache/
      glm52-fp8(新箱先 cp 到 /scratch/models/glm52-fp8,NVMe 装载 13min);serving 环境
      重建 = /sgl-workspace/sglang checkout main 87992eeec + 应用
      /personal/glm52_backup_20260710/patches_main/main_port_full.diff + pip install -e
      python/ --no-deps + sgl-deep-gemm==0.1.4。一切源码以本任务 worktree 为准。
1) 所有 GPU 命令在 rx devbox `glm52-bs1-opt` 上跑(ssh glm52-bs1-opt)。
2) 需要全部 8 卡(单进程多卡 + peer access + cuMulticast workspace);workdir =
   /scratch/kda_bs1/mega01。**不许弄挂常驻 serving**(376.06 基线,重启命令见下)。
3) 编 sm_103a,CUDA 13.0。

背景与禁区(先读 ../../common/prior_art/old02_ar_norm_RESULTS.md):
- 单播 push 路线已判死:36.6µs,~25µs 固定协议成本(world 无关)。不要重做。
- flashinfer 的速度来自 NVLS multimem(multicast st + 交换机内 ld_reduce)——移植时
  必须保留同一协议;workspace/handle 初始化参考 sglang flashinfer_comm_fusion.py 与
  flashinfer python 侧 mnnvl comm 装配代码。
- kernel 源码就在箱上:/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/
  flashinfer/comm/trtllm_mnnvl_allreduce.cuh(纯头文件 JIT 源,复制进 jit_kernel 改造)。
- sglang jit_kernel 模板:参考 python/sglang/jit_kernel/moe_finalize_fuse_shared(我们
  自己的模块,含 load/compile/PDL 用法)。

真实 shape(SHAPES.md,开工先实测复核):T∈{1,6},H=6144,bf16,weight[6144],
eps=1e-5,world=8,融合语义 out=rmsnorm(AR(x)+residual),同时写 residual_out。
**一切 harness、调优、门槛只认 bs=1 真实 shape。**

阶段目标与门槛:
P0 移植(必须先过,不许跳):
  a. jit_kernel 版在 8 卡 harness(照 ../../common/harness/ar_bench.py 的口径:每卡
     50-round CUDA graph 并发 replay)与 flashinfer 原版**逐位一致**(bf16 位级,不是
     rel 容差;两者跑同一 multimem workspace 协议应可位级对齐,若确有不可避免的
     归约序差异,退而求 fp32 oracle rel<1e-3 并说明原因)。
  b. 隔离性能与原版差 ≤3%。
  c. env 开关接入 serving(SGLANG_JIT_MNNVL_AR=1,默认关):打开后 sanity 1×40
     ≥376(不倒退),greedy 输出与基线一致(byte-level diff bench 记录的 text)。
     serving 重启:ssh 后
       cd /scratch/glm52_blog_bench && MODEL_PATH=/scratch/models/glm52-fp8 \
       SGLANG_BS1_BF16_DENSE=1 SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=1 \
       SGLANG_BS1_FP8_DEFER_FINALIZE=1 [SGLANG_JIT_MNNVL_AR=1] \
       nohup setsid ./launch_devbox.sh > server_task01.log 2>&1 & 
     sanity:python3 benchmark_glm52_bs1.py --runs 1 --out-dir results_task01_xxx
P1 bs=1 特化优化(P0 全绿后):
  - 思路池:T=6/H=6144/world=8/eps 全常量化(模板参数);weight 指针跨 replay 稳定
    可预取;grid/block 按 73.7KB payload 重配(现 kernel 按大 payload 设计);fence/
    barrier 粒度(6 token 只需 1 个 flag 轮次?);PDL(cudaGridDependencySynchronize
    入口重叠上一 kernel 尾);把 quant/下一层第一个操作折进 epilogue 的可行性评估。
  - NCU 必须 --replay-mode application(spin/flag kernel 用 kernel-replay 会死锁)。
  - 门槛:T=6 ≤7.0µs(冲 5µs)且 T=1 不劣化;1000-replay 位级稳定(flag/epoch 竞态
    的典型暴露方式);NCU 证据入 RESULTS。
  - e2e promote:sanity ≥378 且输出一致 → 跑官方 3×40 记录。

交付物:jit_kernel 模块(kernel .cuh + python 装配 + workspace 管理)、serving 接入
patch(env 门控,默认关)、RESULTS_SM103.md(P0 位级/性能/e2e 证据 + P1 前后对比 +
NCU 拆解表)、失败路线记录。
```
