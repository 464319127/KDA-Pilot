# Prompt

```
你是 NVIDIA Blackwell/B300(sm_103)elementwise/融合 kernel 专家。任务分两阶段:
P0 把 flashinfer 开源的 RMSNormKernel(csrc/norm.cu)与 RopeQuantizeKernel
(csrc/rope.cu)**原样移植进 sglang jit_kernel**(新模块 bs1_norm_rope_pack),逐位
一致 + 性能持平 + e2e 吞吐不变;P1 针对 bs=1 真实 shape 做融合特化:这批 kernel 单次
只有 2-3µs、payload 只有 KB 级,纯启动延迟型——融合减少 launch 数才是收益来源。
目标合计 0.60ms → ≤0.40ms,e2e −0.2~−0.3 ms/iter。

**NEVER STOP**:持续实现、验证、benchmark、profile、优化,不要问我。

硬性环境约定:
0) 机器断开/过期不是终止条件,按下面处理后继续任务:
   a. rx 传输层掉线(websocket 1006/1012 / Broken pipe)是常态:重连重试;长命令一律
      nohup setsid 后台 + 轮询。
   b. devbox 已过期/被释放:重新申请并重建——
      export PATH=$HOME/.local/bin:$PATH
      rxp devbox acquire --gpu B300 --count 8 --image lmsysorg/sglang:latest --name glm52-bs1-opt
      rxp devbox extend glm52-bs1-opt
      rxp devbox ssh-config glm52-bs1-opt
      ~/.ssh/rx_config 该 Host 的 ProxyCommand 按 bbuf-gb300-8x 样式加 proxychains 包装。
      权重 /cluster-storage/shared/hf_cache/glm52-fp8(先 cp 到 /scratch/models/,NVMe
      装载 13min);serving 环境 = sglang main 87992eeec + /personal/glm52_backup_20260710/
      patches_main/main_port_full.diff + pip install -e python/ --no-deps + sgl-deep-gemm
      0.1.4。一切源码以本任务 worktree 为准。
1) 所有 GPU 命令在 rx devbox `glm52-bs1-opt` 上(ssh glm52-bs1-opt)。
2) kernel 开发用 1 张卡即可(不与 serving 抢卡:先看 nvidia-smi 挑空闲位或错峰);
   e2e 验证需重启 8 卡 serving。workdir = /scratch/kda_bs1/mega02。
3) 编 sm_103a,CUDA 13.0。

素材位置(都在箱上):
- flashinfer JIT 源:/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/
  norm.cu、rope.cu(以及 python 侧 flashinfer/norm.py、rope.py 的装配与 dispatch)。
- sglang jit_kernel 模板:python/sglang/jit_kernel/moe_finalize_fuse_shared(我们自己
  的模块,含编译装配与 PDL 用法)。
- sglang 调用点:layers/layernorm.py(rmsnorm 分发)、models/deepseek_common/
  attention_forward_methods/(MLA rope+quant 段);顺带评估相邻的 sglang 自家小卡
  fused_k_indexer_norm_rope_store / fused_q_indexer_rope_hadamard_quant 是否可并进包。

真实 shape(SHAPES.md,开工先实测复核;全部 bs=1,T∈{1,6}):
- q_a_layernorm [T,2048] 与 kv_a_layernorm [T,512]:**同一 [T,2624] fused_qkv_a 输出的
  相邻切片、同一时刻调用** → 横向融合成 1 launch 是最确定的收益。
- RopeQuantize:q_pe [T,8,64] + k_pe [T,64] → rope → fp8 kv-cache 写入;与前面的 norm
  是紧邻依赖 → 纵向融合候选(norm→rope→quant 一条链)。
- eps=1e-5,bf16 激活,fp8_e4m3 kv-cache。

阶段目标与门槛:
P0 移植(必须先过):
  a. 两个 kernel 的 jit_kernel 版与 flashinfer 原版**逐位一致**(确定性 elementwise,
     必须位级,无容差借口),覆盖 T=1 与 T=6 两组 shape。
  b. 隔离性能 ±3%(单卡 200-round CUDA graph replay 口径)。
  c. env 开关接入 serving(SGLANG_JIT_NORM_ROPE=1,默认关):打开后 sanity 1×40 ≥381,
     greedy 输出与基线逐字一致。
     serving 重启/评测命令同任务 01 prompt(env 换成本任务的)。
P1 融合特化(P0 全绿后):
  - 第一步:q_a+kv_a 双 norm 单 launch(横向);第二步:norm→rope→quant 纵向链;
    T/dims/eps 模板常量化;PDL 入口;评估把 indexer 两小卡并入。
  - 语义与数值:融合后每个站点输出仍须与"分立原版序列"逐位一致(中间量不换精度、
    不换归约序;做不到位级的改动直接放弃该融合)。
  - 门槛:冻结 shape 上融合包 ≥1.4×(对比原版分立总和);1000-replay 稳定;CUDA graph
    可捕获(serving 全在图内,任何 host 逻辑都不许进热路径)。
  - e2e promote:sanity ≥383 且输出一致 → 官方 3×40 记录。

交付物:jit_kernel 模块、serving 接入 patch(env 门控默认关)、RESULTS_SM103.md
(P0/P1 证据 + 每融合步的前后对比表)、失败路线记录。
```
