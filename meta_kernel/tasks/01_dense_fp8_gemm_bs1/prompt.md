# Prompt

```
你是 NVIDIA Blackwell/B300(sm_103)CUDA GEMM kernel 专家。任务:为 GLM-5.2 FP8 bs=1
MTP 解码路径实现 M∈{1,6} 的 w8a8 block-FP8 dense GEMM kernel,在冻结的 production
shape 上按冷 L2 图内口径同时打赢两条基线,达到 promote 门槛后接回 serving。

**NEVER STOP**:持续实现、验证、benchmark、profile、优化,不要问我。遇到问题自己定位。

硬性环境约定:
1) 所有 GPU 命令在 rx devbox `glm52-bs1-opt` 上跑(本机已配 `ssh glm52-bs1-opt` 别名)。
   若 devbox 已过期:`export PATH=$HOME/.local/bin:$PATH && rxp devbox acquire --gpu B300
   --count 8 --image lmsysorg/sglang:latest --name glm52-bs1-opt && rxp devbox ssh-config
   glm52-bs1-opt`,然后把 ~/.ssh/rx_config 里该 Host 的 ProxyCommand 按同文件里
   bbuf-gb300-8x 的样式加上 proxychains 包装(rx 不认代理环境变量)。
2) kernel 微基准固定 GPU7(`K2_DEV=7`);workdir = /scratch/kda_bs1/meta01。
3) devbox pod 本身就是 sglang 0.5.14 容器,nvcc CUDA 13.0,编 sm_103a。
4) 若节点上有 serving 进程在跑,微基准可共存(显存余量 ~47GB/卡),不要杀它。

任务范围(见同目录 SHAPES.md 的完整冻结 shape 表):
1) out[M,N] bf16 = A[M,K] bf16 @ dequant(W[N,K] fp8e4m3, S[N/128,K/128] f32)^T,M∈{1,6}。
2) 双基线,都要在加权 shape 组合上打赢:
   a. DeepGEMM fp8 路径含激活量化(in-graph ~8.7+2.3µs @ M=6);
   b. cuBLAS bf16(现役方案,冷 L2:2624×6144=8.83µs、6144×2048=6.09、512×6144=5.96、
      2048×2048=5.93、6144×256=2.35、3072×6144=9.45、6144×1536=4.91)。
3) M=1 已有胜出移植(decode_gemv.cu,正确性 8/8,6.25-9.91µs)——主交付物是 M=6:
   CUTLASS SM100/103 blockwise-scaled tensor-op 小 M 特化(M pad 到 16)。
4) 先读 PRIOR_ATTEMPTS.md:三条死胡同(Triton×3、手写 mma smem 版、CUDA-core M>1)
   禁止重走;RESULTS_SM103.md 有现状数据。

基准与门槛(不可妥协):
1) 冷 L2 图内口径:≥48 份权重轮转、全部 call 捕进一张 CUDA graph、replay 计时
   (参考 k2_mma_test.py 的 harness;同权重回放测的是 L2 不是 DRAM,禁止)。
2) 正确性:fp32 oracle rel < 2e-2 全 shape + scale-ramp/ragged-N 对抗行;split-K 必须
   确定性(两遍 replay 逐位一致)。
3) promote 门槛:加权 geomean > 1.0 vs 两条基线且无 production 行 < 0.97;NCU 证据
   (ncu-report-skill 规范,spin 类 kernel 用 --replay-mode application)。
4) 达标后接回 serving:hook 在 sglang fp8_utils 的 M≤8 dispatch(见
   ../../integration/README.md),e2e 验证 = /scratch/glm52_blog_bench 的 40 任务×1 轮
   sanity(accept 3.95±0.02、无输出截断),然后 3×40 官方口径记录进 RESULTS_SM103.md。
   接入成功意味着可退役 SGLANG_BS1_BF16_DENSE,运行时回到 100% FP8。

交付物:solution/ 内 kernel + 入口、RESULTS_SM103.md 更新(per-shape 表 + geomean +
NCU 摘要)、serving e2e 前后对比、docs/ 里失败尝试记录。
```
