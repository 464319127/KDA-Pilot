# KDA Prompt: jit_kernel_rope_apply_rope_inplace

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `jit_kernel.rope.apply_rope_inplace`

Goal: optimize or replace this interface for the inclusionAI/Ring-2.6-1T serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `inclusionAI/Ring-2.6-1T`
- Model folder: `llm/ring_26_1t/b200`
- Category: `rope`
- Python interface: `jit_kernel.rope.apply_rope_inplace`
- Captured call count: `44165`
- Captured variants: `592`
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

- `arg[0]: shape=[1, 8, 64], dtype=bfloat16, device=cuda:0, contiguous=False`
- `arg[0]: shape=[1, 8, 64], dtype=bfloat16, device=cuda:1, contiguous=False`
- `arg[0]: shape=[1, 8, 64], dtype=bfloat16, device=cuda:2, contiguous=False`
- `arg[0]: shape=[1, 8, 64], dtype=bfloat16, device=cuda:3, contiguous=False`
- `arg[0]: shape=[1, 8, 64], dtype=bfloat16, device=cuda:4, contiguous=False`
- `arg[0]: shape=[1, 8, 64], dtype=bfloat16, device=cuda:5, contiguous=False`
- `arg[0]: shape=[1, 8, 64], dtype=bfloat16, device=cuda:6, contiguous=False`
- `arg[0]: shape=[1, 8, 64], dtype=bfloat16, device=cuda:7, contiguous=False`
- `arg[0]: shape=[1, 8, 64], dtype=float32, device=cuda:0, contiguous=False`
- `arg[0]: shape=[1, 8, 64], dtype=float32, device=cuda:1, contiguous=False`
- `arg[0]: shape=[1, 8, 64], dtype=float32, device=cuda:2, contiguous=False`
- `arg[0]: shape=[1, 8, 64], dtype=float32, device=cuda:3, contiguous=False`

## Captured Variants

1. label=`random_low`, calls=`282`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 8, 64)\n      dtype=torch.float32\n      device=cuda:7\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[1]=Tensor(\n      shape=(1, 8, 64)\n      dtype=torch.float32\n      device=cuda:7\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(131328, 64)\n      dtype=torch.float32\n      device=cuda:7\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(1,)\n      dtype=torch.int64\n      device=cuda:7\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  is_neox=True\n  rope_dim=0", "scalars": ["is_neox=True", "rope_dim=0"], "tensors": [{"device": "cuda:7", "dtype": "float32", "is_contiguous": false, "kind": "tensor", "name": "arg[0]", "requires_grad": false, ...`
   - kwargs: `{}`
2. label=`random_low`, calls=`280`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 8, 64)\n      dtype=torch.float32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[1]=Tensor(\n      shape=(1, 8, 64)\n      dtype=torch.float32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(131328, 64)\n      dtype=torch.float32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(1,)\n      dtype=torch.int64\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  is_neox=True\n  rope_dim=0", "scalars": ["is_neox=True", "rope_dim=0"], "tensors": [{"device": "cuda:0", "dtype": "float32", "is_contiguous": false, "kind": "tensor", "name": "arg[0]", "requires_grad": false, ...`
   - kwargs: `{}`
3. label=`random_low`, calls=`280`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 8, 64)\n      dtype=torch.float32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[1]=Tensor(\n      shape=(1, 8, 64)\n      dtype=torch.float32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(131328, 64)\n      dtype=torch.float32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(1,)\n      dtype=torch.int64\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  is_neox=True\n  rope_dim=0", "scalars": ["is_neox=True", "rope_dim=0"], "tensors": [{"device": "cuda:1", "dtype": "float32", "is_contiguous": false, "kind": "tensor", "name": "arg[0]", "requires_grad": false, ...`
   - kwargs: `{}`
4. label=`random_low`, calls=`280`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 8, 64)\n      dtype=torch.float32\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[1]=Tensor(\n      shape=(1, 8, 64)\n      dtype=torch.float32\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(131328, 64)\n      dtype=torch.float32\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(1,)\n      dtype=torch.int64\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  is_neox=True\n  rope_dim=0", "scalars": ["is_neox=True", "rope_dim=0"], "tensors": [{"device": "cuda:2", "dtype": "float32", "is_contiguous": false, "kind": "tensor", "name": "arg[0]", "requires_grad": false, ...`
   - kwargs: `{}`
5. label=`random_low`, calls=`280`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 8, 64)\n      dtype=torch.float32\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[1]=Tensor(\n      shape=(1, 8, 64)\n      dtype=torch.float32\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(131328, 64)\n      dtype=torch.float32\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(1,)\n      dtype=torch.int64\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  is_neox=True\n  rope_dim=0", "scalars": ["is_neox=True", "rope_dim=0"], "tensors": [{"device": "cuda:3", "dtype": "float32", "is_contiguous": false, "kind": "tensor", "name": "arg[0]", "requires_grad": false, ...`
   - kwargs: `{}`
6. label=`random_low`, calls=`280`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 8, 64)\n      dtype=torch.float32\n      device=cuda:4\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[1]=Tensor(\n      shape=(1, 8, 64)\n      dtype=torch.float32\n      device=cuda:4\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(131328, 64)\n      dtype=torch.float32\n      device=cuda:4\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(1,)\n      dtype=torch.int64\n      device=cuda:4\n      requires_grad=False\n      is_contiguous=True\n    )\nKeyword input arguments:\n  is_neox=True\n  rope_dim=0", "scalars": ["is_neox=True", "rope_dim=0"], "tensors": [{"device": "cuda:4", "dtype": "float32", "is_contiguous": false, "kind": "tensor", "name": "arg[0]", "requires_grad": false, ...`
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
