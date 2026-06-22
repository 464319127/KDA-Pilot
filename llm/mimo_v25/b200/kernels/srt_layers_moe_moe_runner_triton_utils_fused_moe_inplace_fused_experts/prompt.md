# KDA Prompt: srt_layers_moe_moe_runner_triton_utils_fused_moe_inplace_fused_experts

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts`

Goal: optimize or replace this interface for the XiaomiMiMo/MiMo-V2.5 serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `XiaomiMiMo/MiMo-V2.5`
- Model folder: `llm/mimo_v25/b200`
- Category: `moe`
- Python interface: `srt.layers.moe.moe_runner.triton_utils.fused_moe.inplace_fused_experts`
- Captured call count: `10904`
- Captured variants: `136`
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
- `arg[0]: shape=[1, 4096], dtype=bfloat16, device=cuda:1, contiguous=True`
- `arg[0]: shape=[1, 4096], dtype=bfloat16, device=cuda:2, contiguous=True`
- `arg[0]: shape=[1, 4096], dtype=bfloat16, device=cuda:3, contiguous=True`
- `arg[0]: shape=[103, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[103, 4096], dtype=bfloat16, device=cuda:1, contiguous=True`
- `arg[0]: shape=[103, 4096], dtype=bfloat16, device=cuda:2, contiguous=True`
- `arg[0]: shape=[103, 4096], dtype=bfloat16, device=cuda:3, contiguous=True`
- `arg[0]: shape=[10607, 4096], dtype=bfloat16, device=cuda:0, contiguous=True`
- `arg[0]: shape=[10607, 4096], dtype=bfloat16, device=cuda:1, contiguous=True`
- `arg[0]: shape=[10607, 4096], dtype=bfloat16, device=cuda:2, contiguous=True`
- `arg[0]: shape=[10607, 4096], dtype=bfloat16, device=cuda:3, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`188`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(256, 1024, 4096)\n      dtype=torch.float8_e4m3fn\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(256, 4096, 512)\n      dtype=torch.float8_e4m3fn\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(1, 8)\n      dtype=torch.float32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(1, 8)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=None\n  arg[6]=None\n  arg[7]='silu'\n  arg[8]=True\n  arg[...`
   - kwargs: `{}`
2. label=`random_low`, calls=`188`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(256, 1024, 4096)\n      dtype=torch.float8_e4m3fn\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(256, 4096, 512)\n      dtype=torch.float8_e4m3fn\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(1, 8)\n      dtype=torch.float32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(1, 8)\n      dtype=torch.int32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=None\n  arg[6]=None\n  arg[7]='silu'\n  arg[8]=True\n  arg[...`
   - kwargs: `{}`
3. label=`random_low`, calls=`188`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(256, 1024, 4096)\n      dtype=torch.float8_e4m3fn\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(256, 4096, 512)\n      dtype=torch.float8_e4m3fn\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(1, 8)\n      dtype=torch.float32\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(1, 8)\n      dtype=torch.int32\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=None\n  arg[6]=None\n  arg[7]='silu'\n  arg[8]=True\n  arg[...`
   - kwargs: `{}`
4. label=`random_low`, calls=`188`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 4096)\n      dtype=torch.bfloat16\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(256, 1024, 4096)\n      dtype=torch.float8_e4m3fn\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(256, 4096, 512)\n      dtype=torch.float8_e4m3fn\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(1, 8)\n      dtype=torch.float32\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(1, 8)\n      dtype=torch.int32\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=None\n  arg[6]=None\n  arg[7]='silu'\n  arg[8]=True\n  arg[...`
   - kwargs: `{}`
5. label=`random_low`, calls=`47`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(103, 4096)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(256, 1024, 4096)\n      dtype=torch.float8_e4m3fn\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(256, 4096, 512)\n      dtype=torch.float8_e4m3fn\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(103, 8)\n      dtype=torch.float32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(103, 8)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=None\n  arg[6]=None\n  arg[7]='silu'\n  arg[8]=True\n...`
   - kwargs: `{}`
6. label=`random_low`, calls=`47`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(103, 4096)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(256, 1024, 4096)\n      dtype=torch.float8_e4m3fn\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[2]=Tensor(\n      shape=(256, 4096, 512)\n      dtype=torch.float8_e4m3fn\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(103, 8)\n      dtype=torch.float32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[4]=Tensor(\n      shape=(103, 8)\n      dtype=torch.int32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=None\n  arg[6]=None\n  arg[7]='silu'\n  arg[8]=True\n...`
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
