# KDA Prompt: srt_layers_attention_base_attn_backend_attention_backend_forward

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `srt.layers.attention.base_attn_backend.AttentionBackend.forward`

Goal: optimize or replace this interface for the mistralai/Ministral-3-14B-Instruct-2512 serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `mistralai/Ministral-3-14B-Instruct-2512`
- Model folder: `llm/ministral_3/b200`
- Category: `attention`
- Python interface: `srt.layers.attention.base_attn_backend.AttentionBackend.forward`
- Captured call count: `2287`
- Captured variants: `58`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Workload Coverage

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Shape Summary

- `arg[0]: shape=[1, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[12, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[14685, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[15, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[1542, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[15525, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[16, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[18, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[19, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[24, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[27, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[32, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`40`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([1032], device='cuda:0'), req_pool_indices=tensor([3], device='cuda:0'), seq_lens=tensor([39], device='cuda:0', dtype=\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=RadixAttentio...`
   - kwargs: `{}`
2. label=`random_low`, calls=`40`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([1049], device='cuda:0'), req_pool_indices=tensor([3], device='cuda:0'), seq_lens=tensor([40], device='cuda:0', dtype=\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=RadixAttentio...`
   - kwargs: `{}`
3. label=`random_low`, calls=`40`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([1052], device='cuda:0'), req_pool_indices=tensor([3], device='cuda:0'), seq_lens=tensor([42], device='cuda:0', dtype=\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=RadixAttentio...`
   - kwargs: `{}`
4. label=`random_low`, calls=`40`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([1057], device='cuda:0'), req_pool_indices=tensor([3], device='cuda:0'), seq_lens=tensor([41], device='cuda:0', dtype=\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=RadixAttentio...`
   - kwargs: `{}`
5. label=`random_low`, calls=`40`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([1110], device='cuda:0'), req_pool_indices=tensor([2], device='cuda:0'), seq_lens=tensor([2], device='cuda:0', dtype=t\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=RadixAttentio...`
   - kwargs: `{}`
6. label=`random_low`, calls=`40`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(38, 4096)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(38, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(38, 8, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.EXTEND: 1>, batch_size=1, input_ids=tensor([18746,  1261,  1032,  1049,  1057,  1052,  1048,  1115,  6816,  6947,\n         2314,  1278,  5132, 52165,  1307,  314\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=RadixAt...`
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
