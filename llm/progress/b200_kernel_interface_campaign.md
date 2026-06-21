# B200 LLM Kernel Interface Campaign

Last updated: 2026-06-21T06:39:58Z.

Source baseline:

- SGLang cookbook: `sgl-project/sgl-cookbook@7b5bd9c05e3d86da5161ac2be9b91c19a34bf49d`
- Runtime SGLang checkout: `/data/bbuf/repos/sglang-main`
- Remote B200 node: `cirrascale-gpua83e` / `GPUA83E`
- Capture matrix: `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, `sharegpt_high`
- Evidence source: SGLang kernel Python-interface logging with `SGLANG_KERNEL_API_LOGLEVEL=3`

## Queue

| Requested model | Slug | Status | Cookbook/source mapping | Selected model path | PR |
|---|---|---|---|---|---|
| Qwen3.6 | `qwen_36` | completed; local validation passed | `docs/autoregressive/Qwen/Qwen3.6.md` | `Qwen/Qwen3.6-35B-A3B-FP8` | #55 |
| DeepSeek-V4 | `deepseek_v4` | failed on current B200 env; weight cache cleaned | Present on live docs at `docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4`; absent from cloned cookbook `7b5bd9c`. | `deepseek-ai/DeepSeek-V4-Flash` as single-node B200 primary | pending |
| DeepSeek-V3_2 | `deepseek_v3_2` | completed; local validation passed; weight cache cleaned | `docs/autoregressive/DeepSeek/DeepSeek-V3_2.md` B200 section | `nvidia/DeepSeek-V3.2-NVFP4` | #58 |
| Llama4 | `llama4` | pending | `docs/autoregressive/Llama/Llama4.md`; page covers Scout and Maverick variants. | TBD page variant | pending |
| Gemma4 | `gemma4` | pending | `docs/autoregressive/Google/Gemma4.md`; page covers E2B/E4B/31B/26B-A4B variants. | TBD page variant | pending |
| GPT-OSS | `gpt_oss` | pending | `docs/autoregressive/OpenAI/GPT-OSS.md` | `openai/gpt-oss-120b` | pending |
| Kimi-K2.7-Code | `kimi_k2_7_code` | pending launch validation | Not present in cloned cookbook `7b5bd9c`; HF deployment guide says K2.6/K2.5 launch method is reused. | `moonshotai/Kimi-K2.7-Code` | pending |
| Kimi-K2.6 | `kimi_k2_6` | pending | `docs/autoregressive/Moonshotai/Kimi-K2.6.md` | `moonshotai/Kimi-K2.6` | pending |
| MiniMax-M3 | `minimax_m3` | pending launch validation | Not present in cloned cookbook `7b5bd9c`; HF model exists. | `MiniMaxAI/MiniMax-M3` | pending |
| Nemotron3-Ultra | `nemotron3_ultra` | pending launch validation | Not present in cloned cookbook `7b5bd9c`; HF BF16 and NVFP4 variants exist. | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` | pending |
| Ernie4.5 | `ernie_45` | pending | `docs/autoregressive/Ernie/Ernie4.5.md` | `baidu/ERNIE-4.5-21B-A3B-PT` | pending |
| Step-3.7-Flash | `step_37_flash` | pending launch validation | Not present in cloned cookbook `7b5bd9c`; HF model exists. | `stepfun-ai/Step-3.7-Flash` | pending |
| Ring-2.6-1T | `ring_26_1t` | pending launch validation | Not present in cloned cookbook `7b5bd9c`; HF model exists and is the successor of Ring-2.5-1T cookbook page. | `inclusionAI/Ring-2.6-1T` | pending |
| Intern-S2-Preview | `intern_s2_preview` | pending launch validation | Not present in cloned cookbook `7b5bd9c`; HF model and deployment guide exist. | `internlm/Intern-S2-Preview-FP8` | pending |
| Ministral-3 | `ministral_3` | pending | `docs/autoregressive/Mistral/Ministral-3.md` | `mistralai/Ministral-3-14B-Instruct-2512` | pending |
| MiMo-V2.5 | `mimo_v25` | pending launch validation | Not present in cloned cookbook `7b5bd9c`; HF model exists and follows MiMo-V2-Flash family. | `XiaomiMiMo/MiMo-V2.5` | pending |
| Hunyuan3-Preview | `hunyuan3_preview` | pending via SGLang docs | Not present in cookbook `7b5bd9c`; SGLang docs page `docs/basic_usage/hy3_preview.md`. | `tencent/Hy3-preview` | pending |
| Chroma1.0 | `chroma_10` | special deployment | `docs/autoregressive/FlashLabs/Chroma1.0.md`; hybrid Chroma-SGLang API server, not direct `sglang serve`. | `FlashLabs/Chroma-4B` plus Chroma-SGLang code | pending |

## Completed PRs

- GLM-5.2 reference implementation: PR #53, merged.
- Workload coverage wording fix: PR #54, merged.

## Current Notes

- Qwen3.6 default B200 attention path (`trtllm_mha`) failed during server warmup with a FlashInfer ABI mismatch: `trtllm_paged_attention_decode` expected 27 arguments but got 30. The Qwen3.6 wrapper now uses `--attention-backend triton --disable-flashinfer-autotune` to complete interface shape capture while preserving the cookbook FP8 model path, EAGLE parameters, and B200 `fa4` multimodal attention backend. A follow-up OOM during EAGLE draft torch.compile autotune required lowering `--mem-fraction-static` from `0.8` to `0.7`.
- Qwen3.6 generated 13 kernel interface tasks from 47,479 runtime Python-interface records. All task cards have non-empty direct interface shapes, non-zero calls/variants, and observed coverage for `random_low`, `random_mid`, `random_high`, `sharegpt_low`, `sharegpt_mid`, and `sharegpt_high`. The Qwen3.6 weight cache was manually removed after pulling artifacts.
- DeepSeek-V4 uses the live cookbook Flash B200 command shape: `deepseek-ai/DeepSeek-V4-Flash`, `--tp 4`, `--moe-runner-backend flashinfer_mxfp4`, and `--enable-deepseek-v4-fp4-indexer`. The wrapper also exports `SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16` per the cookbook configuration tip.
- DeepSeek-V4 first launch reached FP4 expert shuffling and DSV4 memory-pool allocation, then failed during FlashInfer autotune dummy warmup with `CUDA error: an illegal memory access was encountered`. The retry keeps the cookbook FP4 MoE/indexer path and adds `--disable-flashinfer-autotune`.
- DeepSeek-V4 retry on GPUs 4-7 with current `sglang-main` (`b4dda8b3ce`) and refreshed editable install reached the first real forward, then failed in FlashInfer 0.6.12 SM100 FP4 MoE with `trtllm_fp4_block_scale_moe` ABI mismatch: first `Expected 35 but got 36 arguments`, then experimental local wrapper hotfix attempts exposed incompatible Python wrapper validation (`AttributeError: 'int' object has no attribute 'dtype'`). `--moe-runner-backend marlin` is not usable on B200 because MXFP4 Marlin requires SM90 or SM120. The FlashInfer core hotfix was reverted, and `/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash` plus logs/capture were removed on GPUA83E.
- DeepSeek-V3_2 is using the current cookbook Blackwell/NVFP4 path: `nvidia/DeepSeek-V3.2-NVFP4`, `--tp 4`, `--quantization modelopt_fp4`, `--moe-runner-backend flashinfer_trtllm`, plus `--tool-call-parser deepseekv32` and `--reasoning-parser deepseek-v3` from the basic usage example. The run is pinned to GPUA83E GPUs 4-7 because GPUs 0-3 still report 100% utilization with 0 MiB memory after the earlier DeepSeek-V4 illegal-memory failure.
- DeepSeek-V3_2 official B200 default DSA backend loaded the full NVFP4 checkpoint successfully (`387G` HF cache, `Load weight end`, ~93.76 GiB/model shard per GPU), then failed during FlashInfer autotune/warmup in DSA decode `trtllm` with `trtllm_paged_attention_decode` ABI mismatch: expected 27 arguments but got 30. The failure signature was observed in the run output before the subsequent retry reset the run directory. Retried with documented DSA fallback `--dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv` while keeping `modelopt_fp4` and `flashinfer_trtllm` MoE.
- DeepSeek-V3_2 `flashmla_kv` retry avoided the TRT-LLM attention ABI mismatch and completed FP4 GEMM autotune, then failed in auto-enabled FlashInfer MNNVL allreduce+rmsnorm fusion with `trtllm_mnnvl_allreduce_fusion` ABI mismatch: expected 15 arguments but got 16. The failure signature was observed in the run output before the subsequent retry reset the run directory. The next retry disabled FlashInfer allreduce fusion with `--enforce-disable-flashinfer-allreduce-fusion` and skipped the slow autotune with `--disable-flashinfer-autotune`.
- DeepSeek-V3_2 retry with DSA `flashmla_kv`, FlashInfer allreduce fusion disabled, and FlashInfer autotune disabled reached real request handling, then failed in `flashinfer_trtllm` FP4 MoE with `trtllm_fp4_block_scale_moe` ABI mismatch: expected 35 arguments but got 36. The failure signature was observed in the run output before the subsequent retry reset the run directory. The successful run switched to the documented `--moe-runner-backend flashinfer_cutlass` fallback while keeping DSA `flashmla_kv`, allreduce fusion disabled, and autotune disabled.
- DeepSeek-V3_2 successful capture used `nvidia/DeepSeek-V3.2-NVFP4`, `--tp 4`, `--quantization modelopt_fp4`, `--moe-runner-backend flashinfer_cutlass`, `--dsa-prefill-backend flashmla_kv`, `--dsa-decode-backend flashmla_kv`, `--kv-cache-dtype fp8_e4m3`, `--enforce-disable-flashinfer-allreduce-fusion`, and `--disable-flashinfer-autotune` on GPUA83E GPUs 4-7. It completed all six workload labels and generated 20 kernel interface tasks from 367,952 runtime Python-interface records. Local validation found no zero-call tasks, no empty shape briefs, and no foreign model names. The `nvidia/DeepSeek-V3.2-NVFP4` HF cache was removed after pulling artifacts.
