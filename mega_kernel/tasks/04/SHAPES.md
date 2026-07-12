# 真实 shape(bs=1 serving,冻结口径)

来源:381.42 官方配置 60 迭代 profile(`/scratch/glm52_blog_bench/profiles/p381/`)+
checkpoint config。**全部 batch_size=1;开工第一步在箱上实测复核(重点:把 attention
bucket 0.99ms/iter 分解到具体 kernel×次数×单价),出入以实测为准并更新本表。**

## 模型/注意力参数(config 实值)

| 参数 | 值 |
|---|---|
| 全局 heads / 每 rank(TP8) | 64 / 8 |
| kv_lora_rank + qk_rope_head_dim | 512 + 64 = 576(MLA 吸收后 MQA 读取宽度) |
| v_head_dim(吸收后输出) | 512(kv_lora) |
| KV cache dtype / page | fp8_e4m3 / page_size 64 |
| DSA index_topk | 2048(indexer 每 token 选 2048 个 kv) |
| indexer | index_n_heads 32,index_head_dim 128(paged MQA logits 走 deepgemm,不在本任务范围) |

## 每 MTP 迭代的注意力调用(实测复核并填全)

| 站点 | 次数/iter | q tokens T | 语义 |
|---|---:|---:|---|
| draft decode(5 步 × 1 层) | ~5 | 1 | 标准 DSA sparse decode;**B200 T=1 wrapper 原生路径直接适用(3.29× 参照)** |
| verify(78 层) | 78 | 6 | TARGET_VERIFY:6 个 q token,各自带 indexer 选出的 topk-2048 页集合 —— **prior kernel 未覆盖的 regime,P1 的主战场** |
| draft-extend(1 层) | ~1 | 6 | DRAFT_EXTEND_V2,类似 verify |
| (indexer paged MQA logits / topk) | — | — | 走 deepgemm/sgl-kernel,**排除在本任务外**(池子归任务 03/未来) |

## 复核方法

1. p381 profile 里把 attention bucket 按 kernel 名分解(`fmhaSm100f*TokenSparse*`、
   `smxx_paged_mqa_logits`、prep 小 kernel 各自 n/iter × µs),写回本表。
2. 在 dsa_backend.py 的 trtllm decode/verify 调用点打印一次真实实参
   (q shape/stride、页表 layout、seq_lens、topk indices dtype int32/int64、scale),
   与 prior/candidate.py 的接口对齐差异列成表。
3. 冻结本表;harness 只认这些 shape。
