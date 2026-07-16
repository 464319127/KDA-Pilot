# 真实 shape(bs=1 serving;开工先实测复核)

MTP draft = 1 层 GlmMoeDsa NextN(eh_proj [6144,12288]→enorm/hnorm、MLA(q_lora 2048/
kv_lora 512/rope 64/heads 8/rank、DSA topk-2048 T=1)、MoE(256 专家 top-8+1 shared,
local 32/rank)、层内 2 次 AR(TP8,mnnvl)),M=1,连续 5 步(步间 token 依赖串行)。
激活全程 [1,*](KB 级,可寄存器/smem 驻留);权重/rank:dense bf16 ~0.6GB + MoE fp8
~151MB + draft KV。@3.5TB/s 纯权重读 ~220µs/步 → 当前 ~240µs/步已近带宽?**开工第一
件事:用 profiler 精确拆 draft 图的 per-step 时长与 launch 间隙,确认压缩空间**
(若 gap+launch 仅 ~50µs/步,目标改为 −0.25ms 并如实汇报)。

复核:profile 60 迭代,draft 图内 kernel 数/步、gap 分布、各段 µs;dump 一步真实输入。
