# KDA Prompt: sgl_kernel_fast_topk_transform_fused

Target GPU: NVIDIA B200.

Target SGLang kernel Python interface to copy as local baseline:

- `sgl_kernel.fast_topk_transform_fused`

Goal: optimize or replace this interface for the nvidia/DeepSeek-V3.2-NVFP4 serving shapes
captured on B200. The shapes below come from runtime SGLang kernel API
logging at the Python interface boundary; they are not torch profiler
CPU-op context shapes.

## Kernel Interface

- Model: `nvidia/DeepSeek-V3.2-NVFP4`
- Model folder: `llm/deepseek_v3_2/b200`
- Category: `sampling`
- Python interface: `sgl_kernel.fast_topk_transform_fused`
- Captured call count: `11224`
- Captured variants: `184`
- Evidence policy: runtime interface capture of args/kwargs/result, not torch-profiler CPU-op shape context.

## Workload Coverage

- `random_low`
- `random_mid`
- `random_high`
- `sharegpt_low`
- `sharegpt_mid`
- `sharegpt_high`

## Shape Summary

- `cu_seqlens_q: shape=[12], dtype=int32, device=cuda:0, contiguous=True`
- `cu_seqlens_q: shape=[12], dtype=int32, device=cuda:1, contiguous=True`
- `cu_seqlens_q: shape=[12], dtype=int32, device=cuda:2, contiguous=True`
- `cu_seqlens_q: shape=[12], dtype=int32, device=cuda:3, contiguous=True`
- `cu_seqlens_q: shape=[13], dtype=int32, device=cuda:0, contiguous=True`
- `cu_seqlens_q: shape=[13], dtype=int32, device=cuda:1, contiguous=True`
- `cu_seqlens_q: shape=[13], dtype=int32, device=cuda:2, contiguous=True`
- `cu_seqlens_q: shape=[13], dtype=int32, device=cuda:3, contiguous=True`
- `cu_seqlens_q: shape=[16], dtype=int32, device=cuda:0, contiguous=True`
- `cu_seqlens_q: shape=[16], dtype=int32, device=cuda:1, contiguous=True`
- `cu_seqlens_q: shape=[16], dtype=int32, device=cuda:2, contiguous=True`
- `cu_seqlens_q: shape=[16], dtype=int32, device=cuda:3, contiguous=True`

## Captured Variants

1. label=`random_low`, calls=`61`
   - args: `[{"kind": "api_arguments", "raw": "Keyword input arguments:\n  score=Tensor(\n      shape=(1, 128)\n      dtype=torch.float32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  lengths=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  page_table_size_1=Tensor(\n      shape=(1, 104)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  cu_seqlens_q=Tensor(\n      shape=(2,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  topk=2048\n  row_starts=None", "scalars": ["topk=2048", "row_starts=None"], "tensors": [{"device": "cuda:0", "dtype": "float32", "is_contiguous": true, "kind": "tensor", "name": "score", "requires_grad": false, "shape": [1, 128]}, {"device...`
   - kwargs: `{}`
2. label=`random_low`, calls=`61`
   - args: `[{"kind": "api_arguments", "raw": "Keyword input arguments:\n  score=Tensor(\n      shape=(1, 128)\n      dtype=torch.float32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  lengths=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  page_table_size_1=Tensor(\n      shape=(1, 105)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  cu_seqlens_q=Tensor(\n      shape=(2,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  topk=2048\n  row_starts=None", "scalars": ["topk=2048", "row_starts=None"], "tensors": [{"device": "cuda:0", "dtype": "float32", "is_contiguous": true, "kind": "tensor", "name": "score", "requires_grad": false, "shape": [1, 128]}, {"device...`
   - kwargs: `{}`
3. label=`random_low`, calls=`61`
   - args: `[{"kind": "api_arguments", "raw": "Keyword input arguments:\n  score=Tensor(\n      shape=(1, 128)\n      dtype=torch.float32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  lengths=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  page_table_size_1=Tensor(\n      shape=(1, 106)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  cu_seqlens_q=Tensor(\n      shape=(2,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  topk=2048\n  row_starts=None", "scalars": ["topk=2048", "row_starts=None"], "tensors": [{"device": "cuda:0", "dtype": "float32", "is_contiguous": true, "kind": "tensor", "name": "score", "requires_grad": false, "shape": [1, 128]}, {"device...`
   - kwargs: `{}`
4. label=`random_low`, calls=`61`
   - args: `[{"kind": "api_arguments", "raw": "Keyword input arguments:\n  score=Tensor(\n      shape=(1, 128)\n      dtype=torch.float32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  lengths=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  page_table_size_1=Tensor(\n      shape=(1, 107)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  cu_seqlens_q=Tensor(\n      shape=(2,)\n      dtype=torch.int32\n      device=cuda:0\n      requires_grad=False\n      is_contiguous=True\n    )\n  topk=2048\n  row_starts=None", "scalars": ["topk=2048", "row_starts=None"], "tensors": [{"device": "cuda:0", "dtype": "float32", "is_contiguous": true, "kind": "tensor", "name": "score", "requires_grad": false, "shape": [1, 128]}, {"device...`
   - kwargs: `{}`
5. label=`random_low`, calls=`61`
   - args: `[{"kind": "api_arguments", "raw": "Keyword input arguments:\n  score=Tensor(\n      shape=(1, 128)\n      dtype=torch.float32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  lengths=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  page_table_size_1=Tensor(\n      shape=(1, 104)\n      dtype=torch.int32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  cu_seqlens_q=Tensor(\n      shape=(2,)\n      dtype=torch.int32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  topk=2048\n  row_starts=None", "scalars": ["topk=2048", "row_starts=None"], "tensors": [{"device": "cuda:1", "dtype": "float32", "is_contiguous": true, "kind": "tensor", "name": "score", "requires_grad": false, "shape": [1, 128]}, {"device...`
   - kwargs: `{}`
6. label=`random_low`, calls=`61`
   - args: `[{"kind": "api_arguments", "raw": "Keyword input arguments:\n  score=Tensor(\n      shape=(1, 128)\n      dtype=torch.float32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  lengths=Tensor(\n      shape=(1,)\n      dtype=torch.int32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  page_table_size_1=Tensor(\n      shape=(1, 105)\n      dtype=torch.int32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  cu_seqlens_q=Tensor(\n      shape=(2,)\n      dtype=torch.int32\n      device=cuda:1\n      requires_grad=False\n      is_contiguous=True\n    )\n  topk=2048\n  row_starts=None", "scalars": ["topk=2048", "row_starts=None"], "tensors": [{"device": "cuda:1", "dtype": "float32", "is_contiguous": true, "kind": "tensor", "name": "score", "requires_grad": false, "shape": [1, 128]}, {"device...`
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
