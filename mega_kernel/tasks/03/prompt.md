# Prompt

```
你是 NVIDIA Blackwell/B300(sm_103)MoE routing kernel 专家。任务分三阶段:
P0 把 flashinfer 开源的 trtllm fused_moe 路由源码(RoutingKernel.cuh 及其依赖头,
routingIndicesDynBlockKernel 所走的 sigmoid noaux-tc top-8 实例化路径)**原样复制**进
sglang jit_kernel(新模块 bs1_moe_routing),用与 flashinfer 相同的模板实参实例化,
对原路由输出逐位一致;P1 **在这份复制的代码上做增量修改**(不是重写!):先删
n_group=1 走不到的 group 阶段死代码,再把 T/E/K 常量化、按 [6,256] 缩 grid、评估单
CTA 化与 PDL——每一步修改后都必须保持位级一致再进下一步,6.6µs → ≤3.5µs;
P2 接回 serving(有明确的集成风险,见下)。
目标 e2e −0.15~−0.25 ms/iter。

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
      权重 /cluster-storage/shared/hf_cache/glm52-fp8(先 cp 到 /scratch/models/);
      serving 环境 = sglang main 87992eeec + /personal/glm52_backup_20260710/patches_main/
      main_port_full.diff + pip install -e python/ --no-deps + sgl-deep-gemm 0.1.4。
1) 所有 GPU 命令在 rx devbox `glm52-bs1-opt` 上(ssh glm52-bs1-opt)。
2) kernel 开发 1 张卡;e2e 需 8 卡 serving。workdir = /scratch/kda_bs1/mega03。
3) 编 sm_103a,CUDA 13.0。

素材位置(都在箱上):
- 路由源码:/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/
  trtllm/fused_moe/RoutingKernel.cuh、RoutingDevKernel.h、RoutingKernelTopK.cuh(如有);
  python 侧装配在 flashinfer 的 fused_moe 模块。
- sglang jit_kernel 模板:python/sglang/jit_kernel/moe_finalize_fuse_shared。
- 调用点:sglang layers/moe/moe_runner/flashinfer_trtllm.py(bypassed 路径把
  routing_logits 交给 wrapper,routing 在其内部启动;routed 路径接受外部 packed topk)。

真实 shape 与路由数学(SHAPES.md,开工先实测复核):[T,256] T∈{1,6},sigmoid 打分,
e_score_correction_bias 参与选择不参与权重,top-8,n_group=1(无 group 阶段——原
kernel 的 group 逻辑直接删),norm_topk_prob=True,routed_scaling_factor=2.5。
另:topk_small_batch(22.6/iter,0.154ms)调用点未定位,开工时实测定位;若与本任务
同属可特化的小 topk,一并纳入;若属 DSA indexer 则记录后排除。

阶段目标与门槛:
P0 语义移植:
  a. 独立 harness:随机 logits + 从 serving 抓的真实 logits 两组,jit 版 topk_ids +
     topk_weights(含 scale/norm 后)与 flashinfer 原路由输出**逐位一致**(ids 完全
     相同;weights 位级,若归约序不可避免则 rel<1e-6 且 argsort 一致,并说明)。
     注意 tie-break 行为要对齐(相同分数的专家排序)。
  b. 隔离性能 ≥ 原版(先不要求加速)。
P1 bs=1 特化:
  - T≤6、E=256、K=8 全常量化:单 CTA(256 专家 = 8 warp × 32 或 2 warp × 128 lane
    评估)、寄存器内 top-8(bitonic/odd-even 小网络)、免 shared-memory 多轮扫描;
    PDL 入口。原 kernel 为大 batch 设计的 DynBlock 机制在 T=6 是纯开销。
  - 门槛:[6,256] ≤3.5µs 且 [1,256] 不劣化;1000-replay 位级稳定;CUDA graph 可捕获。
P2 集成(风险明确,按序尝试):
  - 路径:我们的 routing 输出 → flashinfer PackTopkIds → trtllm_fp8_block_scale_routed_moe。
  - **已知障碍**:serving 的 fp8 defer-finalize(SGLANG_BS1_FP8_DEFER_FINALIZE)当前
    只在 bypassed 格式生效(moe_runner/flashinfer_trtllm.py 里我们自己的 defer 分支
    要求 format_is_bypassed)。集成本 kernel 必须同时把 defer 分支扩展到 routed 格式
    (flashinfer routed 入口同样支持 do_finalize=False,核对后照 bypassed 分支样式加)。
  - env 门控 SGLANG_JIT_MOE_ROUTING=1 默认关;打开后 sanity 1×40 ≥381 且 greedy 输出
    与基线逐字一致(routing 是选择性操作,任何 tie-break 差异都会改输出——位级对齐
    是硬要求);达不到就把集成 park 住,交付独立 kernel + 障碍记录。
  - e2e promote:sanity ≥383 且输出一致 → 官方 3×40 记录。
  serving 重启/评测命令同任务 01 prompt(env 换成本任务的)。

交付物:jit_kernel 模块、(若 P2 成)serving 集成 patch 含 defer 扩展、
RESULTS_SM103.md(P0 位级证据 + P1 前后对比 + NCU + e2e)、失败路线记录。
```
