# Diffusion Kernel × Model Coverage Matrix

Maps each SGLang `python/sglang/jit_kernel/diffusion/` kernel entry point to the
benchmark preset models known to exercise it. This is the cross-task coverage
view used together with the per-task `prompt.md` shape tables.

The matrix is composed of two columns of evidence:

- **Empirical** — observed live during a `kernel_shape_capture.py` sweep over
  the SGLang diffusion benchmark presets on `ion-b200` (B200) and
  `ion8-h200` / `ion9-h200` (H200). The raw JSONL captures land under
  `kernels/<task>/docs/captured_shapes_<arch>.jsonl`.
- **Analytical** — derived from the upstream model architecture configs and
  SGLang code paths but not yet observed live. These models are expected to
  exercise the kernel but the most recent sweep either skipped the preset
  (gated weights, no idle 4-GPU slot, missing local cache, etc.) or the preset
  ran but did not reach the kernel-touching code path within the warmup window.

Re-run the sweep with the remaining presets (FLUX.1-dev, FLUX.2-dev,
HunyuanVideo, MOVA-720p, Helios-Base) when shape evidence for those models is
required for promotion.

## Captured kernels

| Kernel entry point | Empirical models | Notes |
|---|---|---|
| `qknorm_rope.fused_inplace_qknorm_rope` | qwen, zimage (H200) | num_heads in {24, 30}, head_dim=128, rope_dim=128, is_neox=False, eps in {1e-6, 1e-5}; bf16 in/out, fp32 cos_sin_cache, int64 positions |
| `norm_tanh_mul_add_norm_scale.fused_norm_tanh_mul_add` | zimage (H200) | D=3840, S in {4096, 4128}; weight=[D], scale/shift=[1,1,D] |
| `norm_tanh_mul_add_norm_scale.fused_norm_tanh_mul_add_norm_scale` | zimage (H200) | same as above; second-norm-scale variant |
| `rmsnorm_onepass.triton_one_pass_rms_norm` | zimage (H200) | per-head RMS norm tiled, (4096, 128) and (16384, 128) bf16 |
| `scale_residual_norm_scale_shift.fused_norm_scale_shift` | qwen (H200) | D=3072, S in {19, 47, 4096}; norm_type='layer' |
| `scale_residual_norm_scale_shift.fused_scale_residual_norm_scale_shift` | qwen (H200) | residual + gate path, same shapes |
| `scale_shift.fuse_scale_shift_kernel` | qwen (H200) | (B,L,C) = (1, 4096, 3072) / (1, 19, 3072) / (1, 47, 3072) |
| `ltx2_rotary.apply_ltx2_split_rotary_emb` | ltx2 (H200) | inner_dim=2048 (head_dim=64) and inner_dim=4096 (head_dim=128); num_heads=32; half_dim=32 / 64 |

## Not-yet-captured kernels (analytical only)

The kernel families below were not exercised by any preset in the latest sweep.
Their per-task shape tables in `prompt.md` reflect analytical estimates derived
from the upstream model configs. Promotion claims for these kernels require a
new sweep that includes the matching preset.

| Kernel entry point | Analytical models | Why no capture |
|---|---|---|
| `norm.rms_norm_fn` | flux, flux2, hunyuanvideo | gated FLUX weights not pre-downloaded; HunyuanVideo not in HF cache |
| `norm.norm_infer` | qwen, qwen-edit, zimage | exercised by other code paths; not observed in the warmup window of this sweep |
| `group_norm_silu.triton_group_norm_silu` | all VAE decoder paths | the diffusion benchmark perf-dump skips VAE decode by default |
| `group_norm_silu.apply_group_norm_silu` | all VAE decoder paths | same |
| `rotary.apply_rotary_embedding` | wan-t2v, wan-i2v, wan-ti2v, flux, flux2, hunyuanvideo, mova-720p, helios | Wan presets ran but did not hit the wrapped Python entry point; FLUX/Hunyuan/MOVA/Helios not in sweep |
| `scale_shift.fuse_layernorm_scale_shift_gate_select01_kernel` | qwen-edit | qwen-edit subprocess exited before reaching dual-modulation path |
| `scale_shift.fuse_residual_layernorm_scale_shift_gate_select01_kernel` | qwen-edit (residual variant) | same |

## Re-capture procedure

To extend coverage to FLUX / FLUX.2 / HunyuanVideo / MOVA-720p / Helios-Base on
either arch, repeat the sweep from `/root/diffusion_shape_capture/sweep_models.sh`
with the relevant preset list and a valid `HF_TOKEN` for gated repos:

```bash
ssh <host> 'docker exec sglang_bbuf bash -lc "
  HOST_LABEL=<host> ARCH_LABEL=<arch> GPU_LIST_1GPU=<idle-gpu-ids> \
  GPU_LIST_4GPU=<idle-gpu-ids> HF_TOKEN=<token> \
  /root/diffusion_shape_capture/sweep_models.sh \
  flux,flux2,hunyuanvideo,helios /tmp/shapes_<host>.jsonl
"'
```

Then re-run `distribute_shapes.py` (see commit
`Add 16 SGLang diffusion multi-shape kernel KDA tasks`) to merge the new
captures back into each task's `docs/captured_shapes_<arch>.{jsonl,md}` and
`kernels/diffusion_shapes_ledger.md`.
