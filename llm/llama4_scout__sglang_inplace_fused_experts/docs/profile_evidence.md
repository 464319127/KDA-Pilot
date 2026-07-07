# Profile evidence — llama4_scout__sglang_inplace_fused_experts

**Standalone kernel target: 16.7% of total serving GPU time** (max across scenarios) on
`meta-llama/Llama-4-Scout-17B-16E-Instruct`, from the exact cookbook-aligned profile. This is target-selection provenance and headroom context, not the validation path. Clean Python interface (profiler provenance).

> Triton MoE expert-GEMM (sglang's own fused_experts/fused_moe kernel) — single-GPU optimizable; NOT the comm-fused trtllm MoE path (excluded).

- Model: `meta-llama/Llama-4-Scout-17B-16E-Instruct` (slug `llama4_scout`, tp=8)
- Python interface: `sglang.inplace_fused_experts`
- Kernel family: `fused_moe_triton`  ·  Category: `moe`
- GPU kernel(s): `fused_moe_kernel`

## % of GPU time by scenario

| dataset | concurrency | % of GPU time |
|---|---|---|
| random | conc 1 | 7.30% |
| random | conc 32 | 16.70% |
| random | conc 100 | 13.02% |
| sharegpt | conc 1 | 7.34% |
| sharegpt | conc 32 | 10.78% |
| sharegpt | conc 100 | 8.37% |

**Peak: 16.7% in `random_mid` (random, concurrency 32).**

## Input shapes (profiler)
- `[[11912, 5120], [11912, 5120], []]`
- `[[1239, 5120], [16, 2048, 5120], [16, 5120, 1024], [1239, 1], [1239, 1], [], [],`
- `[[1453, 5120], [16, 2048, 5120], [16, 5120, 1024], [1453, 1], [1453, 1], [], [],`
- `[[1891, 5120], [16, 2048, 5120], [16, 5120, 1024], [1891, 1], [1891, 1], [], [],`
- `[[1891, 5120], []]`
- `[[39, 5120], [16, 2048, 5120], [16, 5120, 1024], [39, 1], [39, 1], [], [], [], [`
- `[[5689793], [], [], [], []]`
- `[[5689793], []]`
- `[[9795, 5120], [16, 2048, 5120], [16, 5120, 1024], [9795, 1], [9795, 1], [], [],`
- `[[], [], [], [], [], []]`

## Original serving capture command (provenance only)
```bash
python -m sglang.launch_server --model-path meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8 --trust-remote-code --mem-fraction-static 0.8 --context-length 65536
```
Do not rerun this serving command, `run_capture`, or a multi-GPU e2e A/B as part
of the normal kernel task. Validate with the task-local standalone benchmark on
one idle target GPU using the captured shape set.
