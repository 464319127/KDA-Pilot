# 真实 shape(bs=1 serving,冻结口径)

来源:同任务 01/02(376.06 官方配置 60 迭代 profile)。**全部 batch_size=1;开工第一步
实测复核,出入以实测为准并更新本表。**

## 调用统计(每 MTP 迭代)

| kernel | 次数/iter | 单次 | 合计 |
|---|---:|---:|---:|
| `moe::dev::routing::routingCustom::routingIndicesDynBlockKernel` | 74.7 | 6.6µs | **0.491 ms** |
| `topk_small_batch` | 22.6 | 6.8µs | **0.154 ms** |
| `routingIndicesBlockKernel` | 3.9 | ~4.9µs | 0.019 ms |

74.7 ≈ 75 个 MoE 层(78 层 − first_k_dense_replace=3,verify 图)。topk_small_batch 的
22.6/iter 调用点未定位(候选:draft MTP 层 routing、DSA indexer 相关)——开工时用
torch profiler 调用栈/NVTX 实测定位并记录。

## 路由数学(必须与 checkpoint config 逐项对齐)

| 参数 | 值 |
|---|---|
| n_routed_experts | 256(TP8 下每卡 local 32,local_expert_offset 按 rank) |
| num_experts_per_tok (top-k) | 8 |
| scoring_func | **sigmoid** |
| e_score_correction_bias | 有(noaux-tc:选择用 sigmoid(logits)+bias,权重用不含 bias 的 sigmoid 值) |
| n_group / topk_group | 1 / 1(**无 group 阶段**,kernel 里删掉) |
| norm_topk_prob | True(top-8 权重归一化) |
| routed_scaling_factor | 2.5 |
| routing_logits | [T, 256],T∈{1,6};dtype 开工实测(fp32 或 bf16) |

输出:topk_ids [T,8] + topk_weights [T,8](或 packed 格式,对齐 flashinfer
PackTopkIds/routed wrapper 的消费格式)。

## 复核方法

1. profile 确认两 kernel n/iter 与均值;NCU 看 grid/block(6.6µs 处理 [6,256] 明显
   machinery-bound,记录其 grid 规模作为优化依据)。
2. 从 flashinfer python 侧(fused_moe 装配代码)打印 routing_method_type、bias、
   scale 等实参;与本表核对。
3. 冻结本表;harness 只认这些 shape。
