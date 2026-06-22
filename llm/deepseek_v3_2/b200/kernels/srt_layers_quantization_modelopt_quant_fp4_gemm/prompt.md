# KDA Prompt: srt_layers_quantization_modelopt_quant_fp4_gemm

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `srt.layers.quantization.modelopt_quant.fp4_gemm`

Goal: optimize or replace this interface for the nvidia/DeepSeek-V3.2-NVFP4 serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `nvidia/DeepSeek-V3.2-NVFP4`
- Model folder: `llm/deepseek_v3_2/b200`
- Category: `quant_gemm`
- Python interface: `srt.layers.quantization.modelopt_quant.fp4_gemm`
- Captured call count: `28000`
- Captured variants: `512`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Workload Coverage

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Shape Summary

- `arg[0]: shape=[1, 2048], dtype=uint8, device=cuda:0, contiguous=True`
- `arg[0]: shape=[1, 2048], dtype=uint8, device=cuda:1, contiguous=True`
- `arg[0]: shape=[1, 2048], dtype=uint8, device=cuda:2, contiguous=True`
- `arg[0]: shape=[1, 2048], dtype=uint8, device=cuda:3, contiguous=True`
- `arg[0]: shape=[1, 2304], dtype=uint8, device=cuda:0, contiguous=True`
- `arg[0]: shape=[1, 2304], dtype=uint8, device=cuda:1, contiguous=True`
- `arg[0]: shape=[1, 2304], dtype=uint8, device=cuda:2, contiguous=True`
- `arg[0]: shape=[1, 2304], dtype=uint8, device=cuda:3, contiguous=True`
- `arg[0]: shape=[1, 256], dtype=uint8, device=cuda:0, contiguous=True`
- `arg[0]: shape=[1, 256], dtype=uint8, device=cuda:1, contiguous=True`
- `arg[0]: shape=[1, 256], dtype=uint8, device=cuda:2, contiguous=True`
- `arg[0]: shape=[1, 256], dtype=uint8, device=cuda:3, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`244`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 2048)\n      dtype=torch.uint8\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(2048, 7168)\n      dtype=torch.uint8\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(128, 256)\n      dtype=torch.uint8\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(256, 7168)\n      dtype=torch.float8_e4m3fn\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[4]=Tensor(\n      shape=()\n      dtype=torch.float32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=dtype(\n      repr=torch.bfloat16\n    )\n  arg[6]=7168", "scalars": ["arg[5]=...`
   - kwargs: `{}`
2. label=`random_low`, calls=`244`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 2048)\n      dtype=torch.uint8\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(2048, 7168)\n      dtype=torch.uint8\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(128, 256)\n      dtype=torch.uint8\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(256, 7168)\n      dtype=torch.float8_e4m3fn\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[4]=Tensor(\n      shape=()\n      dtype=torch.float32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=dtype(\n      repr=torch.bfloat16\n    )\n  arg[6]=7168", "scalars": ["arg[5]=...`
   - kwargs: `{}`
3. label=`random_low`, calls=`244`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 2048)\n      dtype=torch.uint8\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(2048, 7168)\n      dtype=torch.uint8\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(128, 256)\n      dtype=torch.uint8\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(256, 7168)\n      dtype=torch.float8_e4m3fn\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[4]=Tensor(\n      shape=()\n      dtype=torch.float32\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=dtype(\n      repr=torch.bfloat16\n    )\n  arg[6]=7168", "scalars": ["arg[5]=...`
   - kwargs: `{}`
4. label=`random_low`, calls=`244`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 2048)\n      dtype=torch.uint8\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(2048, 7168)\n      dtype=torch.uint8\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(128, 256)\n      dtype=torch.uint8\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(256, 7168)\n      dtype=torch.float8_e4m3fn\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[4]=Tensor(\n      shape=()\n      dtype=torch.float32\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=dtype(\n      repr=torch.bfloat16\n    )\n  arg[6]=7168", "scalars": ["arg[5]=...`
   - kwargs: `{}`
5. label=`random_low`, calls=`232`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 256)\n      dtype=torch.uint8\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(256, 7168)\n      dtype=torch.uint8\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(128, 32)\n      dtype=torch.float8_e4m3fn\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(32, 7168)\n      dtype=torch.float8_e4m3fn\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[4]=Tensor(\n      shape=()\n      dtype=torch.float32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=dtype(\n      repr=torch.bfloat16\n    )\n  arg[6]=7168", "scalars": ["arg...`
   - kwargs: `{}`
6. label=`random_low`, calls=`232`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(1, 256)\n      dtype=torch.uint8\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[1]=Tensor(\n      shape=(256, 7168)\n      dtype=torch.uint8\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(128, 32)\n      dtype=torch.float8_e4m3fn\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[3]=Tensor(\n      shape=(32, 7168)\n      dtype=torch.float8_e4m3fn\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[4]=Tensor(\n      shape=()\n      dtype=torch.float32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  arg[5]=dtype(\n      repr=torch.bfloat16\n    )\n  arg[6]=7168", "scalars": ["arg...`
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
