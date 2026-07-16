# 真实 shape(bs=1 serving,冻结口径;开工先在箱上实测复核)

模型:GLM-5.2-FP8,hidden 6144,n_routed_experts 256(TP8 → 每 rank local 32,
local_expert_offset=rank*32),top-8,sigmoid noaux-tc(e_score_correction_bias 参与
选择不参与权重),norm_topk_prob=True,routed_scaling_factor=2.5,n_shared=1,
moe_intermediate 2048(TP8 → 每 rank 256),fp8 128×128 block 量化,激活 bf16。

| 站点 | T | 说明 |
|---|---:|---|
| verify 图(75 个 MoE 层) | 6 | 主战场;routing_logits [6,256],hidden [6,6144] |
| draft/extend(1 层 MTP) | 1 / 6 | 同构小流量 |

链条(当前 5-7 个 kernel,目标 1 个):
routing [T,256]→top8 ids/weights → gemm1: 选中专家 w13 [2×256,6144]fp8 × x_q [T,6144]fp8
→ SiLU-mul [T,256] → gemm2: w2 [6144,256]fp8 → finalize: top-8 加权和 + shared expert
输出加法([T,6144] bf16 输出)。注意 defer-finalize 语义(shared 已在 finalize 融合)。
每层权重字节(local 32 专家)≈ 32×(2×256×6144 + 6144×256)×1B ≈ 151MB → T=6 时纯读
@3.5TB/s ≈ 43µs——**但 top-8 只激活 ≤8/32 local 专家,实际读 ~38MB ≈ 11µs**;当前
gemm 对 39µs 说明离带宽还有 3× 余量,巨核的机会在减少 launch/等待与提高专家级并行。

复核:profile 60 步,按 kernel 名统计链条各环节 n/iter×µs;从 moe_runner 调用点 dump
一次真实 routing_logits/bias/scale。
