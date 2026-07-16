# 真实 shape(bs=1 serving;开工先实测复核)

hidden 6144;fused_qkv_a 权重 [2624,6144] bf16(SGLANG_BS1_BF16_DENSE 生效后;复制式,
每 rank 全量 32MB);输出切片 q_a[T,2048] / kv_a[T,512] / k_pe[T,64];eps=1e-5;
RoPE 64 维 interleaved;KV cache fp8_e4m3(576/条);T∈{1,6}。

链条(当前 5-6 kernel → 目标 1):
GEMM [T,6144]×[6144,2624] → split → RMSNorm(q_a) ∥ RMSNorm(kv_a) → RoPE(k_pe,+q_pe 在
b-path)→ fp8 quant → set_mla_kv_buffer 写入。

字节账:权重 32MB @3.7TB/s ≈ 8.7µs 是硬底;epilogue 各段激活只有 KB 级。当前链
~17-19µs(GEMM 8.7 + 小卡 4×2-3µs + 间隙),融合目标 ≤13µs。

复核:profile 60 步取 n/iter×µs;从 deepseek_common attention forward dump 一次真实
stride/scale/cos-sin cache 形态。
