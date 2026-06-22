# KDA Prompt: jit_kernel_flash_attention_v4_flash_attn_varlen_func

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `jit_kernel.flash_attention_v4.flash_attn_varlen_func`

Goal: optimize or replace this interface for the MiniMaxAI/MiniMax-M3-MXFP8 serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `MiniMaxAI/MiniMax-M3-MXFP8`
- Model folder: `llm/minimax_m3/b200`
- Category: `attention`
- Python interface: `jit_kernel.flash_attention_v4.flash_attn_varlen_func`
- Captured call count: `1656`
- Captured variants: `296`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Workload Coverage

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Shape Summary

- `cu_seqlens_q: shape=[13], dtype=int32, device=cuda:0, contiguous=True`
- `cu_seqlens_q: shape=[13], dtype=int32, device=cuda:1, contiguous=True`
- `cu_seqlens_q: shape=[13], dtype=int32, device=cuda:2, contiguous=True`
- `cu_seqlens_q: shape=[13], dtype=int32, device=cuda:3, contiguous=True`
- `cu_seqlens_q: shape=[13], dtype=int32, device=cuda:4, contiguous=True`
- `cu_seqlens_q: shape=[13], dtype=int32, device=cuda:5, contiguous=True`
- `cu_seqlens_q: shape=[13], dtype=int32, device=cuda:6, contiguous=True`
- `cu_seqlens_q: shape=[13], dtype=int32, device=cuda:7, contiguous=True`
- `cu_seqlens_q: shape=[15], dtype=int32, device=cuda:0, contiguous=True`
- `cu_seqlens_q: shape=[15], dtype=int32, device=cuda:1, contiguous=True`
- `cu_seqlens_q: shape=[15], dtype=int32, device=cuda:2, contiguous=True`
- `cu_seqlens_q: shape=[15], dtype=int32, device=cuda:3, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`12`
   - args: `[{"kind": "api_arguments", "raw": "Keyword input arguments:\n  q=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  k=Tensor(\n      shape=(11428, 128, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  v=Tensor(\n      shape=(11428, 128, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  cu_seqlens_q=Tensor(\n      shape=(2,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  seqused_k=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  max_seqlen_q=1\n  page_table=Tensor(\n      shape=(1, 1)\n      dtype=torch.int32\...`
   - kwargs: `{}`
2. label=`random_low`, calls=`12`
   - args: `[{"kind": "api_arguments", "raw": "Keyword input arguments:\n  q=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  k=Tensor(\n      shape=(11428, 128, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  v=Tensor(\n      shape=(11428, 128, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  cu_seqlens_q=Tensor(\n      shape=(2,)\n      dtype=torch.int32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  seqused_k=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  max_seqlen_q=1\n  page_table=Tensor(\n      shape=(1, 1)\n      dtype=torch.int32\...`
   - kwargs: `{}`
3. label=`random_low`, calls=`12`
   - args: `[{"kind": "api_arguments", "raw": "Keyword input arguments:\n  q=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  k=Tensor(\n      shape=(11428, 128, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  v=Tensor(\n      shape=(11428, 128, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  cu_seqlens_q=Tensor(\n      shape=(2,)\n      dtype=torch.int32\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  seqused_k=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  max_seqlen_q=1\n  page_table=Tensor(\n      shape=(1, 1)\n      dtype=torch.int32\...`
   - kwargs: `{}`
4. label=`random_low`, calls=`12`
   - args: `[{"kind": "api_arguments", "raw": "Keyword input arguments:\n  q=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  k=Tensor(\n      shape=(11428, 128, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  v=Tensor(\n      shape=(11428, 128, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  cu_seqlens_q=Tensor(\n      shape=(2,)\n      dtype=torch.int32\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  seqused_k=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  max_seqlen_q=1\n  page_table=Tensor(\n      shape=(1, 1)\n      dtype=torch.int32\...`
   - kwargs: `{}`
5. label=`random_low`, calls=`12`
   - args: `[{"kind": "api_arguments", "raw": "Keyword input arguments:\n  q=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:4\n      requires_grad=False\n      is_contiguous=True\n    )\n  k=Tensor(\n      shape=(11428, 128, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:4\n      requires_grad=False\n      is_contiguous=True\n    )\n  v=Tensor(\n      shape=(11428, 128, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:4\n      requires_grad=False\n      is_contiguous=True\n    )\n  cu_seqlens_q=Tensor(\n      shape=(2,)\n      dtype=torch.int32\n      device=cuda:4\n      requires_grad=False\n      is_contiguous=True\n    )\n  seqused_k=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:4\n      requires_grad=False\n      is_contiguous=True\n    )\n  max_seqlen_q=1\n  page_table=Tensor(\n      shape=(1, 1)\n      dtype=torch.int32\...`
   - kwargs: `{}`
6. label=`random_low`, calls=`12`
   - args: `[{"kind": "api_arguments", "raw": "Keyword input arguments:\n  q=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:5\n      requires_grad=False\n      is_contiguous=True\n    )\n  k=Tensor(\n      shape=(11428, 128, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:5\n      requires_grad=False\n      is_contiguous=True\n    )\n  v=Tensor(\n      shape=(11428, 128, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:5\n      requires_grad=False\n      is_contiguous=True\n    )\n  cu_seqlens_q=Tensor(\n      shape=(2,)\n      dtype=torch.int32\n      device=cuda:5\n      requires_grad=False\n      is_contiguous=True\n    )\n  seqused_k=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:5\n      requires_grad=False\n      is_contiguous=True\n    )\n  max_seqlen_q=1\n  page_table=Tensor(\n      shape=(1, 1)\n      dtype=torch.int32\...`
   - kwargs: `{}`

Full structured args/kwargs/result records are in `docs/evidence.json`.

## Required First Milestone

1. Copy the upstream SGLang source files needed for this exact interface into `baseline/`.
2. Record upstream URL, commit, and copied files in `docs/baseline_source.md`.
3. Expose the copied baseline through a local low-overhead ABI.
4. Expose the candidate through the exact same ABI in `solution/`.
5. Build correctness tests for every retained captured variant or an explicitly justified representative subset.
6. Benchmark baseline and candidate on an idle B200 with the same shapes, dtypes, devices, contiguity, and scalar parameters.
- Unsupported shapes or parameter combinations must fall back to the recovered SGLang baseline.

Do not import, patch, or monkey-patch a live SGLang server during correctness or benchmark runs.
