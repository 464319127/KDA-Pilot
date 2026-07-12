## GLM-5.2 unified attention(DSA/MLA 稀疏 decode)— B200 per-regime 加速比 (2026-07-08)

baseline = 已装 flashinfer 0.6.12(`fmhaSm100f` trtllm-gen);candidate = 本任务 native-CUDA 两级 split-KV flash-decode。加速比 = baseline 中位数 / candidate 中位数(CUDA-event,warmup 10 / 7 trials,B200 空闲 GPU4);54/54 correctness 通过。

| regime(function, batch/topk) | n | baseline µs | candidate µs | 是否 fallback | 加速比 |
|---|---:|---:|---:|:---:|---:|
| decode B=1 短序列 (random_low_short, tk2048) | 16 | 24.0–26.3 | ~12.35 | 否 (native) | 1.95–2.13× |
| decode B=1 长序列 L=2048 (sharegpt_long, tk2048) | 8 | 23.3–25.8 | ~14.4 | 否 (native) | 1.62–1.79× |
| decode B=1 短 seq_lens 回归 (33/100/1000) | 3 | 23.5–25.5 | 13.4–14.4 | 否 (native) | 1.63–1.89× |
| decode B=16 (tk2048) | 4 | 22.2–22.6 | 22.3–22.6 | 是 (baseline) | 0.99–1.01× |
| decode 大 batch B=19/285/2701 (tk2048) | 3 | 24 / 148 / 1104 | ≈ | 是 (baseline) | 0.98–0.99× |
| decode 回归 (b1 tk64, b2 tk128) | 2 | 24.1–25.1 | ≈ | 是 (baseline) | 1.00–1.02× |
| DSA wrapper forward_decode/_forward_trtllm (全 B) | 9 | 91–599 | ≈ | 是 (baseline) | 0.98–1.02× |
| forward_extend (全 B) | 4 | 20–598 | ≈ | 是 (baseline) | 1.00–1.01× |
| ragged prefill trtllm_ragged (B=8/16/113) | 3 | 17.6–20.0 | ≈ | 是 (baseline) | 0.99–1.02× |
| dense MHA _forward_standard_mha (B=8/113) | 2 | 17.9–19.3 | ≈ | 是 (baseline) | 1.00× |

**几何平均:** headline(34 production)**1.564×**;原生 B=1 decode(27 行)**1.874×**(1.62–2.13×);其余全部 baseline fallback 持平(~1.0×,无回退)。正确性 54/54。


## per_token_group_quant (B200) — NO-GO

| shape (M×N) | 格式 | 加速比 (native候选/baseline) | fallback(提升态) |
|---|---|---|---|
| 2701×6144 | ue8m0 | 0.984× | 是 (baseline) |
| 2701×2048 | ue8m0 | 0.759× | 是 (baseline) |
| 2701×1536 | ue8m0 | 0.999× | 是 (baseline) |
## GLM-5.2 unified attention(DSA/MLA 稀疏 decode)— Fable 5 重跑 — PROMOTED (2026-07-09)

baseline = 已装 flashinfer 0.6.12(`fmhaSm100f` trtllm-gen);candidate = native-CUDA 两级 split-KV flash-decode(移植自上节 Opus 版内核并全量复测)+ **新增 T=1 wrapper 全原生路径**(单 kernel 融合 strided gather + 交错 RoPE + fp8 量化 + 当前 token KV 写入,再接原生 decode core,替换 baseline 的 5 launch 前处理序列)。加速比 = baseline 中位数 / candidate 中位数(CUDA-event,warmup 10 / 7 trials,inner-loop 放大至 ~1000µs,隔离子进程,B200 空闲 GPU4);51/51 correctness 通过(46 冻结 production + 5 回归,不重打标签;wrapper 行含 KV-pool 原位写入校验)。

| regime(function, batch/topk) | n | baseline µs | candidate µs | 是否 fallback | 加速比 |
|---|---:|---:|---:|:---:|---:|
| decode B=1 短序列 (random_low_short, tk2048) | 16 | 23.8–27.2 | ~12.36 | 否 (native) | 1.93–2.19× |
| decode B=1 长序列 L=2048 (sharegpt_long, tk2048) | 8 | 23.4–25.8 | ~14.41 | 否 (native) | 1.62–1.79× |
| decode B=1 短 seq_lens 回归 (17/33/1000) | 3 | 23.2–23.7 | 12.4–14.4 | 否 (native) | 1.65–1.89× |
| decode B=1 seq_len=2048 边界回归 | 1 | 23.0 | 14.4 | 否 (native) | 1.60× |
| DSA wrapper T=1 forward_decode/_forward_trtllm | 4 | 73.5–76.6 | ~22.7 | 否 (native, 融合 prep) | **3.25–3.37×** |
| decode B=16 (tk2048, 含短seq回归) | 5 | 22.2–25.5 | ≈ | 是 (baseline) | 0.99–1.00× |
| decode 大 batch B=19/285/2701 (tk2048) | 3 | 23.6 / 148 / 1092 | ≈ | 是 (baseline) | 0.99–1.00× |
| DSA wrapper T=16/19/285/2701 | 5 | 70.7–1187.6 | ≈ | 是 (baseline) | 1.01–1.03× |
| forward_extend (T=19/113/285/2701) | 4 | 17.9–1191.3 | ≈ | 是 (baseline) | 0.99–1.02× |
| ragged prefill trtllm_ragged (T=113) | 1 | 17.9 | ≈ | 是 (baseline) | 1.01× |
| dense MHA _forward_standard_mha (T=113) | 1 | 17.9 | ≈ | 是 (baseline) | 0.99× |

**几何平均:** headline(46 冻结 production)**1.5414×** → **PROMOTED**;原生 B=1 decode(28 行含 4 回归)**1.85×**(production 24 行 **1.874×**,1.60–2.19×);**T=1 wrapper 原生(4 行)3.293×**(3.25–3.37×);其余全部 baseline fallback 持平(19 行 geomean 1.0027×,0.99–1.03×,无回退)。正确性 51/51(另 9/9 独立 oracle 含 RoPE-vs-flashinfer 契约用例;claimed 路径 runtime kernel-name purity trace 全净)。

对比注记:① headline 分母口径不同(本次 46 冻结 production vs 上节 Opus 重标签后 34 production),两个 headline 数字不可直接比;换算同口径粗估:按 Opus 34 行口径本次 ≈1.79×(24×1.874 + 4×3.293 + 6×≈1.0),按本次 46 行冻结口径 Opus 版 ≈1.39×(24×1.874 + 22×≈1.0)。② 同 regime 对齐:原生 B=1 decode 两次完全一致(production geomean 均 **1.874×**,同一 split-KV 设计的移植复测);Fable 增量 = T=1 wrapper 全原生路径 **3.25–3.37×**(Opus 版该 regime fallback ~1.0×)。③ B/T=16/19 扩展为实测 no-go(B=16/19 leaf 0.19–0.24×、T=16/19 full-wrapper 0.58–0.68×、B=16 短 seq 0.46×:SIMT gather 内核在 ≥256K 次散射 576B 读取下 DRAM 延迟受限,cubin 自 0.865 waves 起摊销固定开销),非纯分析推断。④ 修复一个基准共享陷阱:trtllm-gen cubin 在 launch 间于 workspace_buffer 保留活跃状态,native 内核改用私有 scratch(否则交错 A/B 在 B=16 长序列下第二次 baseline launch 非法访存)。



