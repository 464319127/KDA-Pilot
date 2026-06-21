# KDA Prompt: srt_layers_attention_base_attn_backend_attention_backend_forward

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `srt.layers.attention.base_attn_backend.AttentionBackend.forward`

Goal: optimize or replace this interface for the nvidia/DeepSeek-V3.2-NVFP4 serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `nvidia/DeepSeek-V3.2-NVFP4`
- Model folder: `llm/deepseek_v3_2/b200`
- Category: `attention`
- Python interface: `srt.layers.attention.base_attn_backend.AttentionBackend.forward`
- Captured call count: `13664`
- Captured variants: `224`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Executed Workload Matrix

The capture run executed all workload labels below for this model.
A specific interface may still be absent from a workload when the
serving path does not call it for that dataset/concurrency level.

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Observed Workloads For This Interface

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Not Observed For This Interface

- none

## Shape Summary

- `arg[0]: shape=[1, 32, 512], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[1, 32, 512], dtype=bfloat16, device=cuda:1, contiguous=True`
- `arg[0]: shape=[1, 32, 512], dtype=bfloat16, device=cuda:2, contiguous=True`
- `arg[0]: shape=[1, 32, 512], dtype=bfloat16, device=cuda:3, contiguous=True`
- `arg[0]: shape=[10139, 32, 192], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[10139, 32, 192], dtype=bfloat16, device=cuda:1, contiguous=True`
- `arg[0]: shape=[10139, 32, 192], dtype=bfloat16, device=cuda:2, contiguous=True`
- `arg[0]: shape=[10139, 32, 192], dtype=bfloat16, device=cuda:3, contiguous=True`
- `arg[0]: shape=[103, 32, 192], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[103, 32, 192], dtype=bfloat16, device=cuda:1, contiguous=True`
- `arg[0]: shape=[103, 32, 192], dtype=bfloat16, device=cuda:2, contiguous=True`
- `arg[0]: shape=[103, 32, 192], dtype=bfloat16, device=cuda:3, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`61`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 32, 512)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 1, 512)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 1, 512)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([551], device='cuda:0'), req_pool_indices=tensor([3], device='cuda:0'), seq_lens=tensor([105], device='cuda:0'), out_c\n    )\n  arg[5]=True\nKeyword input arguments:\n  q_rope=Tensor(\n      shape...`
   - kwargs: `{}`
2. label=`random_low`, calls=`61`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 32, 512)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 1, 512)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 1, 512)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([60037], device='cuda:0'), req_pool_indices=tensor([3], device='cuda:0'), seq_lens=tensor([106], device='cuda:0'), out\n    )\n  arg[5]=True\nKeyword input arguments:\n  q_rope=Tensor(\n      shape...`
   - kwargs: `{}`
3. label=`random_low`, calls=`61`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 32, 512)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 1, 512)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 1, 512)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([64583], device='cuda:0'), req_pool_indices=tensor([3], device='cuda:0'), seq_lens=tensor([107], device='cuda:0'), out\n    )\n  arg[5]=True\nKeyword input arguments:\n  q_rope=Tensor(\n      shape...`
   - kwargs: `{}`
4. label=`random_low`, calls=`61`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 32, 512)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 1, 512)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 1, 512)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([9294], device='cuda:0'), req_pool_indices=tensor([3], device='cuda:0'), seq_lens=tensor([104], device='cuda:0'), out_\n    )\n  arg[5]=True\nKeyword input arguments:\n  q_rope=Tensor(\n      shape...`
   - kwargs: `{}`
5. label=`random_low`, calls=`61`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 32, 512)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 1, 512)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 1, 512)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([551], device='cuda:1'), req_pool_indices=tensor([3], device='cuda:1'), seq_lens=tensor([105], device='cuda:1'), out_c\n    )\n  arg[5]=True\nKeyword input arguments:\n  q_rope=Tensor(\n      shape...`
   - kwargs: `{}`
6. label=`random_low`, calls=`61`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 32, 512)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 1, 512)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 1, 512)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([60037], device='cuda:1'), req_pool_indices=tensor([3], device='cuda:1'), seq_lens=tensor([106], device='cuda:1'), out\n    )\n  arg[5]=True\nKeyword input arguments:\n  q_rope=Tensor(\n      shape...`
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
