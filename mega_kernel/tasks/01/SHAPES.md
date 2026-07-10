# 真实 shape(bs=1 serving,冻结口径)

来源:GLM-5.2-FP8 8×B300 TP8,EAGLE 5-1-6,bs=1,376.06 tok/s 官方配置的 torch profiler
(`/scratch/glm52_blog_bench/profiles/full376/`,60 迭代)。**所有 shape 必须是 batch_size=1
serving 的真实值;开工第一步在箱上实测复核本表,任何出入以实测为准并更新本表。**

## 调用统计(每 MTP 迭代)

| kernel | 次数/iter | 单次耗时 | 合计 |
|---|---:|---:|---:|
| `oneshotAllreduceFusionKernel` | 157.3 | 8.3µs | **1.309 ms** |

157 ≈ 78 层 × 2(attn-out AR、mlp-out AR,verify 图)+ draft/extend 图内少量。

## 张量形状(bf16,hidden = 6144,world = 8)

| 场景 | tokens T | payload | 备注 |
|---|---:|---|---|
| verify 图(占主) | 6 | [6, 6144] bf16 = 73.7 KB | num_draft_tokens=6 |
| draft 图(5 步 M=1) | 1 | [1, 6144] bf16 = 12.3 KB | 1 层 MTP 头 |
| draft-extend 图 | 6 | [6, 6144] | 同 verify |

融合语义(与 flashinfer oneshot fused 一致,不许减功能):
`out = rmsnorm(allreduce(x) + residual, weight, eps=1e-5)`,同时输出 `residual_out = allreduce(x) + residual`。
weight [6144] bf16;eps=1e-5(config rms_norm_eps)。

## 复核方法

1. 在 376 服务器上抓 60 步 profile(命令见 prompt),统计该 kernel 的 n/iter 与均值。
2. 在 sglang 调用点(`flashinfer_comm_fusion.py` / mnnvl comm 路径)临时打印一次
   shape/dtype/eps/flags 后立刻移除。
3. 冻结进本表后,harness 与 promote 门槛只认本表 shape。
