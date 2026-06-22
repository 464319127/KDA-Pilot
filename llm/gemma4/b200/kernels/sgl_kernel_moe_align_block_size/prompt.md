# KDA Prompt: sgl_kernel_moe_align_block_size

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `sgl_kernel.moe_align_block_size`

Goal: optimize or replace this interface for the google/gemma-4-26B-A4B-it serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `google/gemma-4-26B-A4B-it`
- Model folder: `llm/gemma4/b200`
- Category: `moe`
- Python interface: `sgl_kernel.moe_align_block_size`
- Captured call count: `1650`
- Captured variants: `31`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Workload Coverage

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Shape Summary

- `arg[0]: shape=[1, 8], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[103, 8], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[11, 8], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[12, 8], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[12673, 8], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[15, 8], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[17204, 8], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[21, 8], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[22, 8], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[26, 8], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[28, 8], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[3, 8], dtype=int32, device=cuda:0, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`120`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 8)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=129\n  arg[2]=16\n  arg[3]=Tensor(\n      shape=(128,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(8,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[6]=Tensor(\n      shape=(130,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[7]=True", "scalars": ["arg[1]=129", "arg[2]=16", "arg[7]=True"], "tensors": [{"devic...`
   - kwargs: `{}`
2. label=`random_low`, calls=`30`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(103, 8)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=129\n  arg[2]=16\n  arg[3]=Tensor(\n      shape=(2759,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(173,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[6]=Tensor(\n      shape=(130,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[7]=True", "scalars": ["arg[1]=129", "arg[2]=16", "arg[7]=True"], "tensors": [{"...`
   - kwargs: `{}`
3. label=`random_mid`, calls=`30`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 8)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=129\n  arg[2]=16\n  arg[3]=Tensor(\n      shape=(128,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(8,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[6]=Tensor(\n      shape=(130,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[7]=True", "scalars": ["arg[1]=129", "arg[2]=16", "arg[7]=True"], "tensors": [{"devic...`
   - kwargs: `{}`
4. label=`random_mid`, calls=`30`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(12, 8)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=129\n  arg[2]=16\n  arg[3]=Tensor(\n      shape=(1536,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(96,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[6]=Tensor(\n      shape=(130,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[7]=True", "scalars": ["arg[1]=129", "arg[2]=16", "arg[7]=True"], "tensors": [{"de...`
   - kwargs: `{}`
5. label=`random_mid`, calls=`30`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(12673, 8)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=129\n  arg[2]=256\n  arg[3]=Tensor(\n      shape=(134279,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(525,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[6]=Tensor(\n      shape=(130,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[7]=True", "scalars": ["arg[1]=129", "arg[2]=256", "arg[7]=True"], "tensors...`
   - kwargs: `{}`
6. label=`random_mid`, calls=`30`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(15, 8)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=129\n  arg[2]=16\n  arg[3]=Tensor(\n      shape=(1920,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(120,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[6]=Tensor(\n      shape=(130,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[7]=True", "scalars": ["arg[1]=129", "arg[2]=16", "arg[7]=True"], "tensors": [{"d...`
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
