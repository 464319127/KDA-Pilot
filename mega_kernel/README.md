# mega_kernel — GLM-5.2-FP8 bs=1 解码 kernel 优化战役(8×B300 / sm_103)

目标:GLM-5.2-FP8 在 8×B300(TP8)上 bs=1 MTP(EAGLE 5-1-6)解码吞吐从基线
307.17 持续推进到 **≥400 tok/s**(官方口径 = 3 轮 × 40 任务共 120 请求,报
mean decode tok/s)。当前已达 **381.42**(serving 侧 patch + 任务 01),上游 PR:
[#30957](https://github.com/sgl-project/sglang/pull/30957)(bf16-dense)
[#30958](https://github.com/sgl-project/sglang/pull/30958)(FP8 defer-finalize)
[#30959](https://github.com/sgl-project/sglang/pull/30959)(mnnvl AR jit 特化)。

## 一次 MTP 迭代的 kernel 占比(torch profiler,60 迭代,381.42 配置实测)

一次迭代 ≈ 10.1ms = draft 图(5×M=1)+ verify 图(M=6 过 78 层)+ draft-extend 图
(M=6 过 1 层)。双流乒乓执行,union busy 9.67ms,暴露气泡仅 ~0.4ms。
kernel 时长总和口径(两大流合计,含重叠,sum=12.7ms):

| 类别 | ms/iter | 占比 | 来源 | 可改? |
|---|---:|---:|---|---|
| dense GEMM bf16(nvjet+splitK reduce) | 4.70 | 37% | cuBLAS 闭源 | 换源码实现才可改 |
| MoE GEMM(gemm1+gemm2) | 2.87 | 23% | trtllm-gen **cubin** | 换源码实现才可改 |
| MoE 辅助(routing/finalize/act/topk) | 1.75 | 14% | flashinfer JIT 源 + sglang 自家 | ✓ |
| 融合 AR(+add+rmsnorm)×157 | 1.22 | 10% | flashinfer JIT 源 | ✓(任务 01 已做) |
| attention(DSA sparse,topk-2048) | 0.99 | 8% | trtllm-gen cubin(核心)+ 源码前处理 | 部分 ✓ |
| norm/rope/quant/elementwise | 1.20 | 9% | flashinfer JIT 源 + aten | ✓ |

**关键教训(选题前必读)**:双流重叠区里的 kernel 砍了不缩关键路径(隔离 1.1× ≠
e2e 收益);**全 rank 同步点(AR)和串行链上的大块(GEMM/attention)才 1:1 转化**。
立项先用 `LEARNINGS.md` 里的双流 union 归因法确认目标在关键路径上。

## 任务板

| 任务 | 对象 | 池子 ms/iter | 状态 |
|---|---|---:|---|
| [01](tasks/01/) | mnnvl 融合 AR+add+rmsnorm jit 移植 + bs=1 特化 | 1.31 | ✅ **完成并采纳**:单次 8.3→7.6µs,e2e 376.06→**381.42** 官方(+1.4%),位级一致;上游 PR #30959 |
| [02](tasks/02/) | RMSNorm + RopeQuantize 移植与融合包 | 0.60 | 开放 |
| [03](tasks/03/) | MoE sigmoid top-8/256 routing + small-batch topk | 0.64 | 开放 |
| [04](tasks/04/) | DSA sparse decode 原生化(cubin 替换,B200 有 1.874×/3.29× 参照内核) | 0.99 | 开放 |
| [09](tasks/09/) | **bs=1 MoE 巨核**:routing→gemm1→SiLU→gemm2→finalize(+shared)单 persistent kernel,替换 trtllm-gen cubin 链 | 4.6(GEMM 2.87+aux 1.75) | 开放,难度最高/收益最大 |
| [10](tasks/10/) | **MLA a-path 竖向融合链**:fused_qkv_a GEMM 尾接 双RMSNorm→RoPE→fp8-quant→KV写 单 kernel | ~0.9(关键路径) | 开放 |
| [11](tasks/11/) | **draft-step 巨核**(TileRT 风格):MTP 1 层 M=1 前向整层单 kernel,5 步驻留 | ~1.2(draft 图) | 开放 |

(编号 05-08 属于并行的 K3/GB300 子战役——`tasks/05-08`,Kimi-K3 bs=1,另见其各自 config;本表为 GLM-5.2/B300 主线。)

一键启动:`scripts/launch_kernels/k0X_*.sh`(自动建 worktree + Claude Code + RLCR 循环)。

## 铁律

0. **能增量就不重写**:有源码参照的(flashinfer JIT / B200 已晋升内核)从移植开始,
   每步位级一致;巨核类任务(05/06/07)属于结构性重写,正确性口径改为 fp32 oracle
   rel ≤2e-2 + e2e 质量门(sanity 不倒退 + accept ≥3.80 + 官方 3×40)。
1. **shape 全部来自 bs=1 真实 serving**(T∈{1,6},EAGLE 5-1-6 配置冻结),开工先在箱
   上实测复核 SHAPES.md。
2. serving 接入一律 env 门控、默认关;基线 **381.42** 随时可复现(重建流程见各 prompt
   第 0 条,~1 小时)。
3. promote = 隔离达标 + 关键路径归因证明可转化 + e2e sanity ≥383 + 官方 3×40 记录。
4. 机器断开/过期不是终止(rx devbox 重申请流程写在各 prompt 第 0 条)。

## 环境与资产

- 机器:rx devbox 池 8×B300(`rxp devbox acquire --gpu B300 --count 8 --image lmsysorg/sglang:latest`)。
- serving 基线:sglang main `87992eeec` + `/personal/glm52_backup_20260710/patches_main/main_port_full.diff` + 任务01接入器;权重 NVMe 拷贝后重启 ~13 分钟。
- 评测:`benchmark_glm52_bs1.py`(sanity `--runs 1` / 官方 `--runs 3`);profile 用 /start_profile `num_steps≤100`。
- `LEARNINGS.md`:判死方向(fp8 小 M dense、自研 unicast AR、重叠区小卡)与基准协议(cold-L2 48 份、application-replay NCU、1000-replay 位级稳定)。
