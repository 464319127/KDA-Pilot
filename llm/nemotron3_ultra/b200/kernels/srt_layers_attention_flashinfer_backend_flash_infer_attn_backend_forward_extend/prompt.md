# KDA Prompt: srt_layers_attention_flashinfer_backend_flash_infer_attn_backend_forward_extend

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `srt.layers.attention.flashinfer_backend.FlashInferAttnBackend.forward_extend`

Goal: optimize or replace this interface for the nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16`
- Model folder: `llm/nemotron3_ultra/b200`
- Category: `attention`
- Python interface: `srt.layers.attention.flashinfer_backend.FlashInferAttnBackend.forward_extend`
- Captured call count: `1344`
- Captured variants: `112`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Workload Coverage

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Shape Summary

- `arg[0]: shape=[103, 1024], dtype=bfloat16, device=cuda:0, contiguous=False`
- `arg[0]: shape=[103, 1024], dtype=bfloat16, device=cuda:1, contiguous=False`
- `arg[0]: shape=[103, 1024], dtype=bfloat16, device=cuda:2, contiguous=False`
- `arg[0]: shape=[103, 1024], dtype=bfloat16, device=cuda:3, contiguous=False`
- `arg[0]: shape=[103, 1024], dtype=bfloat16, device=cuda:4, contiguous=False`
- `arg[0]: shape=[103, 1024], dtype=bfloat16, device=cuda:5, contiguous=False`
- `arg[0]: shape=[103, 1024], dtype=bfloat16, device=cuda:6, contiguous=False`
- `arg[0]: shape=[103, 1024], dtype=bfloat16, device=cuda:7, contiguous=False`
- `arg[0]: shape=[161, 1024], dtype=bfloat16, device=cuda:0, contiguous=False`
- `arg[0]: shape=[161, 1024], dtype=bfloat16, device=cuda:1, contiguous=False`
- `arg[0]: shape=[161, 1024], dtype=bfloat16, device=cuda:2, contiguous=False`
- `arg[0]: shape=[161, 1024], dtype=bfloat16, device=cuda:3, contiguous=False`

## Captured Variants

1. label=`random_low`, calls=`12`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(103, 1024)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[1]=Tensor(\n      shape=(103, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(103, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.EXTEND: 1>, batch_size=1, input_ids=tensor([ 1784,  1659,  1053,  1048,  1048, 10033,  1307, 20592,  9551, 41101,\n         1593,  1044,  2725, 20999, 20791,  142\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=Rad...`
   - kwargs: `{}`
2. label=`random_low`, calls=`12`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(103, 1024)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[1]=Tensor(\n      shape=(103, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(103, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.EXTEND: 1>, batch_size=1, input_ids=tensor([ 1784,  1659,  1053,  1048,  1048, 10033,  1307, 20592,  9551, 41101,\n         1593,  1044,  2725, 20999, 20791,  142\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=Rad...`
   - kwargs: `{}`
3. label=`random_low`, calls=`12`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(103, 1024)\n      dtype=torch.bfloat16\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[1]=Tensor(\n      shape=(103, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(103, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:2\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.EXTEND: 1>, batch_size=1, input_ids=tensor([ 1784,  1659,  1053,  1048,  1048, 10033,  1307, 20592,  9551, 41101,\n         1593,  1044,  2725, 20999, 20791,  142\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=Rad...`
   - kwargs: `{}`
4. label=`random_low`, calls=`12`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(103, 1024)\n      dtype=torch.bfloat16\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[1]=Tensor(\n      shape=(103, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(103, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:3\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.EXTEND: 1>, batch_size=1, input_ids=tensor([ 1784,  1659,  1053,  1048,  1048, 10033,  1307, 20592,  9551, 41101,\n         1593,  1044,  2725, 20999, 20791,  142\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=Rad...`
   - kwargs: `{}`
5. label=`random_low`, calls=`12`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(103, 1024)\n      dtype=torch.bfloat16\n      device=cuda:4\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[1]=Tensor(\n      shape=(103, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:4\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(103, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:4\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.EXTEND: 1>, batch_size=1, input_ids=tensor([ 1784,  1659,  1053,  1048,  1048, 10033,  1307, 20592,  9551, 41101,\n         1593,  1044,  2725, 20999, 20791,  142\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=Rad...`
   - kwargs: `{}`
6. label=`random_low`, calls=`12`
   - args: `[{"kind": "api_arguments", "raw": "Positional input arguments:\n  arg[0]=Tensor(\n      shape=(103, 1024)\n      dtype=torch.bfloat16\n      device=cuda:5\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[1]=Tensor(\n      shape=(103, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:5\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[2]=Tensor(\n      shape=(103, 1, 128)\n      dtype=torch.bfloat16\n      device=cuda:5\n      requires_grad=False\n      is_contiguous=False\n    )\n  arg[3]=RadixAttention(\n      repr=RadixAttention()\n    )\n  arg[4]=ForwardBatch(\n      repr=ForwardBatch(forward_mode=<ForwardMode.EXTEND: 1>, batch_size=1, input_ids=tensor([ 1784,  1659,  1053,  1048,  1048, 10033,  1307, 20592,  9551, 41101,\n         1593,  1044,  2725, 20999, 20791,  142\n    )\n  arg[5]=True", "scalars": ["arg[3]=RadixAttention(", "repr=Rad...`
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
