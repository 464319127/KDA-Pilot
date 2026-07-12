# Prompt

```
你是 NVIDIA Blackwell/B300(sm_103)attention kernel 专家。任务:把 B200 上已晋升的
原生 DSA/MLA 稀疏 decode kernel(prior/ 里的 mla_sparse_decode.cu 两级 split-KV core +
mla_wrapper_prep.cu T=1 融合前处理,B200 实测 decode B=1 tk2048 1.874×、T=1 wrapper
3.29×,51/51 正确性)移植到 sm103 并接入 GLM-5.2-FP8 bs=1 serving,替换 trtllm-gen
fmhaSm100f TokenSparse cubin。attention 在关键路径上占 0.99ms/iter,目标 e2e
−0.25~−0.45 ms/iter(381.42 → ~392-400)。

**NEVER STOP**:持续实现、验证、benchmark、profile、优化,不要问我。

**本任务的正确性口径(与任务 01-03 不同,owner 已批准)**:flash-decode 的归约序与
cubin 不同,**不要求位级一致**。门槛改为:fp32 oracle rel ≤ 2e-2(B200 任务同款)+
真实 serving 张量上与 cubin 输出交叉校验(rel ≤ 2e-2)+ e2e 质量门
(sanity ≥385 且 accept_len ≥3.80 且 40/40 任务输出完整合理 + 官方 3×40 ≥384)。

硬性环境约定:
0) 机器断开/过期不是终止条件,按下面处理后继续任务:
   a. rx 传输层掉线(websocket 1006/1012 / Broken pipe)是常态:重连重试;长命令一律
      nohup setsid 后台 + 轮询。
   b. devbox 已过期/被释放:重新申请并重建——
      export PATH=$HOME/.local/bin:$PATH
      rxp devbox acquire --gpu B300 --count 8 --image lmsysorg/sglang:latest --name glm52-bs1-opt
      rxp devbox extend glm52-bs1-opt
      rxp devbox ssh-config glm52-bs1-opt
      ~/.ssh/rx_config 该 Host 的 ProxyCommand 照同文件 bbuf-gb300-8x 的样式补
      `env -u http_proxy ... proxychains4 -q -f ~/.config/rx/proxychains.conf` 包装。
      权重 /cluster-storage/shared/hf_cache/glm52-fp8(先 cp 到 /scratch/models/,断点
      续传脚本照 /personal/glm52_backup_20260710 里的 copy_weights.sh);serving 环境 =
      sglang main 87992eeec(浅克隆需先 git fetch origin main)+ /personal/
      glm52_backup_20260710/patches_main/main_port_full.diff + pip install -e python/
      --no-deps + sgl-deep-gemm==0.1.4 + 任务01接入器(KDA-Pilot main
      mega_kernel/tasks/01/solution/serving/sglang_patcher.py apply)。
1) 所有 GPU 命令在 rx devbox `glm52-bs1-opt` 上(ssh glm52-bs1-opt)。
2) kernel 开发 1 张卡;e2e 需 8 卡重启 serving(基线 381.42,五个 env flag 见下)。
   workdir = /scratch/kda_bs1/mega04。
3) 编 sm_103a,CUDA 13.0。

serving 基线(不许弄挂;e2e 前后都要能复现):
  cd /scratch/glm52_blog_bench && MODEL_PATH=/scratch/models/glm52-fp8 \
  SGLANG_BS1_BF16_DENSE=1 SGLANG_ENABLE_MOE_DEFERRED_FINALIZE=1 \
  SGLANG_BS1_FP8_DEFER_FINALIZE=1 SGLANG_JIT_MNNVL_AR=1 SGLANG_JIT_MNNVL_AR_OPT=1 \
  [SGLANG_JIT_DSA_DECODE=1] nohup setsid ./launch_devbox.sh > server_task04.log 2>&1 &
  sanity: python3 benchmark_glm52_bs1.py --runs 1 --out-dir results_task04_xxx
  profile: /start_profile num_steps≤100(严禁不限步,曾把服务器 OOM 打挂)

背景与禁区(先读 ../../LEARNINGS.md + prior/B200_RESULTS_EXCERPT.md):
- flag 级后端替换已判死:--dsa-decode-backend flashmla_auto 在 forward_extend 直接
  ValueError(2026-07-12 实测);fa3/tilelang 对 spec 三态覆盖同样不全,不要再试 flag。
- prior kernel 的已知边界(B200 实测):B≥16 / T≥16 native 是 no-go(SIMT gather 在
  ≥256K 次 576B 散射读下 DRAM 延迟受限)——bs=1 的 T∈{1,6} 在甜区;大 batch 一律
  fallback cubin,不要试图扩覆盖。
- 基准陷阱(B200 踩过):trtllm-gen cubin 会在 workspace_buffer 里跨 launch 保留活跃
  状态——A/B 交错基准时 native 侧必须用私有 scratch,否则第二次 baseline launch 非法
  访存。
- NCU 对 spin/协作 kernel 用 --replay-mode application。

阶段目标与门槛:
P0 sm103 移植(modify-in-place,不许重写):
  a. prior/ 三件套编译到 sm_103a,先跑 B200 任务的冻结 shape 正确性集(fp32 oracle
     rel ≤2e-2);再从 live serving 抓真实张量(q/页表/topk indices/scale)回放,
     与 trtllm cubin 输出交叉校验 rel ≤2e-2。
  b. 隔离基准:T=1 decode tk2048 对 cubin ≥1.3×(B200 是 1.87×,sm103 允许衰减;
     低于 1.3× 先 NCU 找原因再进 P1)。
P1 覆盖扩展 + sm103 特化(主战场):
  - **T=6 TARGET_VERIFY / DRAFT_EXTEND_V2 是 prior 未覆盖的 regime,也是 serving
    价值大头(78 次/iter)**:每个 q token 有独立的 topk-2048 页集合;扩展 core 的
    q 维并保持 split-KV 结构(6 个 token 可按 head×token 平铺 CTA)。
  - T=1 wrapper 融合前处理路径(3.29× 参照)对 draft 5 步直接适用,一并接。
  - sm103 retune:SM 数/L2 与 B200 不同,split 因子、CTA 形态、每级归约宽度重扫;
    fp8 KV 读取路径确认走 LDG.128/TMA 最优形态。
  - 门槛:T=1 ≥1.5×、T=6 ≥1.3×(vs cubin 同 shape),oracle+交叉校验全绿,
    1000-replay 稳定,CUDA graph 可捕获(三个图:draft/verify/extend 都要能捕)。
P2 接入 serving:
  - dsa_backend.py 新增 dsa_impl "jit_native"(env SGLANG_JIT_DSA_DECODE=1 默认关):
    DECODE / TARGET_VERIFY / DRAFT_EXTEND_V2 走新 kernel,prefill 与未覆盖形态
    fallback trtllm(参考 dsa_impl 分发结构;flashmla 的失败就是 extend 没接,引以为戒)。
  - e2e:sanity ≥385 且 accept ≥3.80 且输出质量正常(40/40 完整、mean tokens 与基线
    同量级)→ 官方 3×40 ≥384 记录进 RESULTS。达不到就 park,交付 kernel + 数据。

交付物:jit_kernel 模块(bs1_dsa_decode)、dsa_backend 接入 patch(env 门控默认关)、
RESULTS_SM103.md(P0 移植证据 + P1 per-regime 对比表 + NCU + e2e)、失败路线记录。
```
