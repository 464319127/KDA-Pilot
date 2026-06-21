# KDA Prompt: sgl_kernel_moe_align_block_size

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `sgl_kernel.moe_align_block_size`

Goal: optimize or replace this interface for the baidu/ERNIE-4.5-21B-A3B-PT serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `baidu/ERNIE-4.5-21B-A3B-PT`
- Model folder: `llm/ernie_45/b200`
- Category: `moe`
- Python interface: `sgl_kernel.moe_align_block_size`
- Captured call count: `1530`
- Captured variants: `31`
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

- `arg[0]: shape=[1, 6], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[10124, 6], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[12, 6], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[14708, 6], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[15, 6], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[1543, 6], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[15580, 6], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[16, 6], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[18, 6], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[19, 6], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[24, 6], dtype=int32, device=cuda:0, contiguous=True`
- `arg[0]: shape=[27, 6], dtype=int32, device=cuda:0, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`126`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 6)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=65\n  arg[2]=16\n  arg[3]=Tensor(\n      shape=(96,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(6,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[6]=Tensor(\n      shape=(66,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[7]=True", "scalars": ["arg[1]=65", "arg[2]=16", "arg[7]=True"], "tensors": [{"device": ...`
   - kwargs: `{}`
2. label=`random_low`, calls=`27`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(38, 6)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=65\n  arg[2]=16\n  arg[3]=Tensor(\n      shape=(1203,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(76,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[6]=Tensor(\n      shape=(66,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[7]=True", "scalars": ["arg[1]=65", "arg[2]=16", "arg[7]=True"], "tensors": [{"devic...`
   - kwargs: `{}`
3. label=`random_mid`, calls=`54`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(27, 6)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=65\n  arg[2]=16\n  arg[3]=Tensor(\n      shape=(1137,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(72,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[6]=Tensor(\n      shape=(66,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[7]=True", "scalars": ["arg[1]=65", "arg[2]=16", "arg[7]=True"], "tensors": [{"devic...`
   - kwargs: `{}`
4. label=`random_mid`, calls=`27`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(12, 6)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=65\n  arg[2]=16\n  arg[3]=Tensor(\n      shape=(1047,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(66,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[6]=Tensor(\n      shape=(66,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[7]=True", "scalars": ["arg[1]=65", "arg[2]=16", "arg[7]=True"], "tensors": [{"devic...`
   - kwargs: `{}`
5. label=`random_mid`, calls=`27`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(15, 6)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=65\n  arg[2]=16\n  arg[3]=Tensor(\n      shape=(1065,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(67,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[6]=Tensor(\n      shape=(66,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[7]=True", "scalars": ["arg[1]=65", "arg[2]=16", "arg[7]=True"], "tensors": [{"devic...`
   - kwargs: `{}`
6. label=`random_mid`, calls=`27`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1543, 6)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=65\n  arg[2]=128\n  arg[3]=Tensor(\n      shape=(17513,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(137,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[6]=Tensor(\n      shape=(66,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[7]=True", "scalars": ["arg[1]=65", "arg[2]=128", "arg[7]=True"], "tensors": [{...`
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
