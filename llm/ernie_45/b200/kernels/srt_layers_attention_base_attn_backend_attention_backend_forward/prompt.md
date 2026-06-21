# KDA Prompt: srt_layers_attention_base_attn_backend_attention_backend_forward

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `srt.layers.attention.base_attn_backend.AttentionBackend.forward`

Goal: optimize or replace this interface for the baidu/ERNIE-4.5-21B-A3B-PT serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `baidu/ERNIE-4.5-21B-A3B-PT`
- Model folder: `llm/ernie_45/b200`
- Category: `attention`
- Python interface: `srt.layers.attention.base_attn_backend.AttentionBackend.forward`
- Captured call count: `1586`
- Captured variants: `57`
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

- `arg[0]: shape=[1, 2560], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[10124, 2560], dtype=bfloat16, device=cuda:0, contiguous=False`
- `arg[0]: shape=[12, 2560], dtype=bfloat16, device=cuda:0, contiguous=False`
- `arg[0]: shape=[14708, 2560], dtype=bfloat16, device=cuda:0, contiguous=False`
- `arg[0]: shape=[15, 2560], dtype=bfloat16, device=cuda:0, contiguous=False`
- `arg[0]: shape=[1543, 2560], dtype=bfloat16, device=cuda:0, contiguous=False`
- `arg[0]: shape=[15580, 2560], dtype=bfloat16, device=cuda:0, contiguous=False`
- `arg[0]: shape=[16, 2560], dtype=bfloat16, device=cuda:0, contiguous=False`
- `arg[0]: shape=[18, 2560], dtype=bfloat16, device=cuda:0, contiguous=False`
- `arg[0]: shape=[19, 2560], dtype=bfloat16, device=cuda:0, contiguous=False`
- `arg[0]: shape=[24, 2560], dtype=bfloat16, device=cuda:0, contiguous=False`
- `arg[0]: shape=[27, 2560], dtype=bfloat16, device=cuda:0, contiguous=False`

## Captured Variants

1. label=`random_low`, calls=`28`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 4, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 4, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([12], device='cuda:0'), req_pool_indices=tensor([3], device='cuda:0'), seq_lens=tensor([41], device='cuda:0', dtype=to\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=RadixAttentio...`
   - kwargs: `{}`
2. label=`random_low`, calls=`28`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 4, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 4, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([4], device='cuda:0'), req_pool_indices=tensor([3], device='cuda:0'), seq_lens=tensor([40], device='cuda:0', dtype=tor\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=RadixAttentio...`
   - kwargs: `{}`
3. label=`random_low`, calls=`28`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 4, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 4, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([7], device='cuda:0'), req_pool_indices=tensor([3], device='cuda:0'), seq_lens=tensor([42], device='cuda:0', dtype=tor\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=RadixAttentio...`
   - kwargs: `{}`
4. label=`random_low`, calls=`28`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 4, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 4, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([93919], device='cuda:0'), req_pool_indices=tensor([3], device='cuda:0'), seq_lens=tensor([39], device='cuda:0', dtype\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=RadixAttentio...`
   - kwargs: `{}`
5. label=`random_low`, calls=`28`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(38, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[1]=Tensor(\n      shape=(38, 4, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(38, 4, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.EXTEND: 1>, batch_size=1, input_ids=tensor([ 5209,   274, 93919,     4,    12,     7,     3, 93927,  5874,  6968,\n         1071,   290,  3243, 22078,   315,  190\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=RadixA...`
   - kwargs: `{}`
6. label=`random_low`, calls=`18`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 2560)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(1, 4, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(1, 4, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.DECODE: 2>, batch_size=1, input_ids=tensor([3], device='cuda:0'), req_pool_indices=tensor([2], device='cuda:0'), seq_lens=tensor([2], device='cuda:0', dtype=torc\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=RadixAttentio...`
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
