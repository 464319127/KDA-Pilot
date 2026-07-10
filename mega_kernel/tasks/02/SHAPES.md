# 真实 shape(bs=1 serving,冻结口径)

来源:同任务 01(376.06 官方配置 60 迭代 profile)。**全部 batch_size=1;开工第一步
实测复核,出入以实测为准并更新本表。**

## 调用统计(每 MTP 迭代)

| kernel | 次数/iter | 单次 | 合计 |
|---|---:|---:|---:|
| `flashinfer RMSNormKernel` | 169.1 | 2.2µs | **0.372 ms** |
| `flashinfer RopeQuantizeKernel` | 81.6 | 2.8µs | **0.227 ms** |
| (相邻小卡:`fused_k_indexer_norm_rope_store` 22.6×2.7µs、`fused_q_indexer_rope_hadamard_quant` 22.6×2.3µs — sglang 自家,可顺带纳入融合评估) | | | 0.112 ms |

169 ≈ 78 层 ×2(q_a_layernorm [T,2048] + kv_a_layernorm [T,512],verify T=6)+ draft
(1 层 ×2×5 步,T=1)+ 头尾杂项。81.6 ≈ 78 层 ×1(verify)+ draft。

## 张量形状(模型参数:hidden 6144,q_lora 2048,kv_lora 512,rope 64,
heads/rank 8(TP8),qk_nope 192,eps 1e-5,kv-cache fp8_e4m3)

| 站点 | 输入 | 输出 | 说明 |
|---|---|---|---|
| q_a_layernorm | [T, 2048] bf16(fused_qkv_a 输出切片) | [T, 2048] bf16 | T∈{1,6} |
| kv_a_layernorm | [T, 512] bf16(同一 [T,2624] 输出的相邻切片) | [T, 512] bf16 | 与上同源同刻 → 横向融合首选 |
| RopeQuantize | q_pe [T, 8, 64] + k_pe [T, 64] bf16(+cos/sin cache) | rope 后写 fp8 kv-cache 条目 | 精确签名开工时从 flashinfer rope.cu 与 sglang MLA 调用点(deepseek_common attention forward)实测确认 |

## 复核方法

1. profile 60 步,确认两 kernel 的 n/iter、均值、grid/block(NCU 或 trace args)。
2. 在 sglang 调用点(layernorm.py 的 flashinfer rmsnorm 分发、MLA forward 的
   rope+quant 段)临时打印 shape/dtype/eps/rope 参数,记录后移除。
3. 冻结本表;harness 只认这些 shape。
