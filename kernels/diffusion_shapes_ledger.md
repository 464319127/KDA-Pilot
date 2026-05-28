# Diffusion Shape Ledger (captured from SGLang benchmark presets)

This file is the cross-task summary of shapes captured from the SGLang
diffusion benchmark sweep, aggregated across `ion-b200`, `ion8-h200`, and
`ion9-h200`. The per-task `docs/captured_shapes_<arch>.md` files carry the
same data scoped to one kernel family.

## `group_norm_silu.apply_group_norm_silu`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| h200 | hunyuanvideo | `[1, 512, 5, 32, 32]/float16C` | `<GroupNorm>` ; `<SiLU>` |
| h200 | hunyuanvideo | `[1, 128, 17, 256, 256]/float16C` | `<GroupNorm>` ; `<SiLU>` |
| h200 | hunyuanvideo | `[1, 256, 9, 128, 128]/float16C` | `<GroupNorm>` ; `<SiLU>` |
| h200 | hunyuanvideo | `[1, 512, 5, 24, 64]/float16C` | `<GroupNorm>` ; `<SiLU>` |
| h200 | hunyuanvideo | `[1, 512, 5, 12, 10]/float16NC` | `<GroupNorm>` ; `<SiLU>` |
| h200 | hunyuanvideo | `[1, 256, 17, 256, 256]/float16C` | `<GroupNorm>` ; `<SiLU>` |
| h200 | hunyuanvideo | `[1, 256, 9, 48, 128]/float16NC` | `<GroupNorm>` ; `<SiLU>` |
| h200 | hunyuanvideo | `[1, 512, 2, 64, 20]/float16C` | `<GroupNorm>` ; `<SiLU>` |
| h200 | hunyuanvideo | `[1, 512, 2, 12, 32]/float16NC` | `<GroupNorm>` ; `<SiLU>` |
| h200 | hunyuanvideo | `[1, 256, 9, 128, 40]/float16NC` | `<GroupNorm>` ; `<SiLU>` |
| h200 | hunyuanvideo | `[1, 512, 9, 128, 128]/float16C` | `<GroupNorm>` ; `<SiLU>` |
| h200 | hunyuanvideo | `[1, 256, 3, 128, 128]/float16C` | `<GroupNorm>` ; `<SiLU>` |
| h200 | hunyuanvideo | `[1, 512, 2, 24, 64]/float16NC` | `<GroupNorm>` ; `<SiLU>` |

## `group_norm_silu.triton_group_norm_silu`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| h200 | hunyuanvideo | `[1, 512, 5, 32, 32]/float16C` ; `[512]/float16C` ; `[512]/float16C` | num_groups=`32` ; eps=`1e-06` |
| h200 | hunyuanvideo | `[1, 128, 17, 256, 256]/float16C` ; `[128]/float16C` ; `[128]/float16C` | num_groups=`32` ; eps=`1e-06` |
| h200 | hunyuanvideo | `[1, 256, 9, 128, 128]/float16C` ; `[256]/float16C` ; `[256]/float16C` | num_groups=`32` ; eps=`1e-06` |
| h200 | hunyuanvideo | `[1, 512, 5, 24, 64]/float16C` ; `[512]/float16C` ; `[512]/float16C` | num_groups=`32` ; eps=`1e-06` |
| h200 | hunyuanvideo | `[1, 512, 5, 12, 10]/float16NC` ; `[512]/float16C` ; `[512]/float16C` | num_groups=`32` ; eps=`1e-06` |
| h200 | hunyuanvideo | `[1, 256, 17, 256, 256]/float16C` ; `[256]/float16C` ; `[256]/float16C` | num_groups=`32` ; eps=`1e-06` |
| h200 | hunyuanvideo | `[1, 256, 9, 48, 128]/float16NC` ; `[256]/float16C` ; `[256]/float16C` | num_groups=`32` ; eps=`1e-06` |
| h200 | hunyuanvideo | `[1, 512, 2, 64, 20]/float16C` ; `[512]/float16C` ; `[512]/float16C` | num_groups=`32` ; eps=`1e-06` |
| h200 | hunyuanvideo | `[1, 512, 2, 12, 32]/float16NC` ; `[512]/float16C` ; `[512]/float16C` | num_groups=`32` ; eps=`1e-06` |
| h200 | hunyuanvideo | `[1, 256, 9, 128, 40]/float16NC` ; `[256]/float16C` ; `[256]/float16C` | num_groups=`32` ; eps=`1e-06` |
| h200 | hunyuanvideo | `[1, 512, 9, 128, 128]/float16C` ; `[512]/float16C` ; `[512]/float16C` | num_groups=`32` ; eps=`1e-06` |
| h200 | hunyuanvideo | `[1, 256, 3, 128, 128]/float16C` ; `[256]/float16C` ; `[256]/float16C` | num_groups=`32` ; eps=`1e-06` |
| h200 | hunyuanvideo | `[1, 512, 2, 24, 64]/float16NC` ; `[512]/float16C` ; `[512]/float16C` | num_groups=`32` ; eps=`1e-06` |

## `ltx2_rotary.apply_ltx2_split_rotary_emb`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| h200 | ltx2 | `[2, 6144, 4096]/bfloat16C` ; `[2, 32, 6144, 64]/bfloat16NC` ; `[2, 32, 6144, 64]/bfloat16NC` |  |
| h200 | ltx2 | `[2, 126, 2048]/bfloat16C` ; `[2, 32, 126, 32]/bfloat16NC` ; `[2, 32, 126, 32]/bfloat16NC` |  |
| h200 | ltx2 | `[2, 6144, 2048]/bfloat16C` ; `[2, 32, 6144, 32]/bfloat16NC` ; `[2, 32, 6144, 32]/bfloat16NC` |  |
| h200 | ltx2 | `[1, 24576, 2048]/bfloat16C` ; `[1, 32, 24576, 32]/bfloat16NC` ; `[1, 32, 24576, 32]/bfloat16NC` |  |

## `norm.norm_infer`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| h200 | helios | `[8640, 5120]/float32C` ; `[5120]/float32C` ; `[5120]/float32C` | eps=`1e-06` ; is_rms_norm=`False` |

## `norm_tanh_mul_add_norm_scale.fused_norm_tanh_mul_add`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| b200 | zimage | `[1, 4096, 3840]/bfloat16C` ; `[3840]/bfloat16C` ; `[1, 1, 3840]/bfloat16C` ; `[1, 4096, 3840]/bfloat16C` | `None` ; `rms` ; `1e-05` |
| b200 | zimage | `[1, 4128, 3840]/bfloat16C` ; `[3840]/bfloat16C` ; `[1, 1, 3840]/bfloat16C` ; `[1, 4128, 3840]/bfloat16C` | `None` ; `rms` ; `1e-05` |
| h200 | zimage | `[1, 4096, 3840]/bfloat16C` ; `[3840]/bfloat16C` ; `[1, 1, 3840]/bfloat16C` ; `[1, 4096, 3840]/bfloat16C` | `None` ; `rms` ; `1e-05` |
| h200 | zimage | `[1, 4128, 3840]/bfloat16C` ; `[3840]/bfloat16C` ; `[1, 1, 3840]/bfloat16C` ; `[1, 4128, 3840]/bfloat16C` | `None` ; `rms` ; `1e-05` |

## `norm_tanh_mul_add_norm_scale.fused_norm_tanh_mul_add_norm_scale`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| b200 | zimage | `[1, 4096, 3840]/bfloat16C` ; `[3840]/bfloat16C` ; `[1, 1, 3840]/bfloat16C` ; `[1, 4096, 3840]/bfloat16C` ; `[3840]/bfloat16C` ; `[1, 1, 3840]/bfloat16C` | `None` ; `None` ; `rms` ; `1e-05` |
| b200 | zimage | `[1, 4128, 3840]/bfloat16C` ; `[3840]/bfloat16C` ; `[1, 1, 3840]/bfloat16C` ; `[1, 4128, 3840]/bfloat16C` ; `[3840]/bfloat16C` ; `[1, 1, 3840]/bfloat16C` | `None` ; `None` ; `rms` ; `1e-05` |
| h200 | zimage | `[1, 4096, 3840]/bfloat16C` ; `[3840]/bfloat16C` ; `[1, 1, 3840]/bfloat16C` ; `[1, 4096, 3840]/bfloat16C` ; `[3840]/bfloat16C` ; `[1, 1, 3840]/bfloat16C` | `None` ; `None` ; `rms` ; `1e-05` |
| h200 | zimage | `[1, 4128, 3840]/bfloat16C` ; `[3840]/bfloat16C` ; `[1, 1, 3840]/bfloat16C` ; `[1, 4128, 3840]/bfloat16C` ; `[3840]/bfloat16C` ; `[1, 1, 3840]/bfloat16C` | `None` ; `None` ; `rms` ; `1e-05` |

## `qknorm_rope.fused_inplace_qknorm_rope`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| b200 | qwen-edit | q=`[8424, 24, 128]/bfloat16C` ; k=`[8424, 24, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[8424, 128]/float32C` ; positions=`[8424]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |
| b200 | qwen-edit | q=`[195, 24, 128]/bfloat16C` ; k=`[195, 24, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[195, 128]/float32C` ; positions=`[195]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |
| b200 | qwen-edit | q=`[189, 24, 128]/bfloat16C` ; k=`[189, 24, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[189, 128]/float32C` ; positions=`[189]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |
| b200 | zimage | q=`[4096, 30, 128]/bfloat16C` ; k=`[4096, 30, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[4096, 128]/float32C` ; positions=`[4096]/int64C` | is_neox=`False` ; eps=`1e-05` ; head_dim=`128` ; rope_dim=`128` |
| b200 | zimage | q=`[32, 30, 128]/bfloat16C` ; k=`[32, 30, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[32, 128]/float32C` ; positions=`[32]/int64C` | is_neox=`False` ; eps=`1e-05` ; head_dim=`128` ; rope_dim=`128` |
| b200 | zimage | q=`[4128, 30, 128]/bfloat16C` ; k=`[4128, 30, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[4128, 128]/float32C` ; positions=`[4128]/int64C` | is_neox=`False` ; eps=`1e-05` ; head_dim=`128` ; rope_dim=`128` |
| h200 | qwen | q=`[4096, 24, 128]/bfloat16C` ; k=`[4096, 24, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[4096, 128]/float32C` ; positions=`[4096]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |
| h200 | qwen | q=`[19, 24, 128]/bfloat16C` ; k=`[19, 24, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[19, 128]/float32C` ; positions=`[19]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |
| h200 | qwen | q=`[47, 24, 128]/bfloat16C` ; k=`[47, 24, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[47, 128]/float32C` ; positions=`[47]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |
| h200 | zimage | q=`[4096, 30, 128]/bfloat16C` ; k=`[4096, 30, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[4096, 128]/float32C` ; positions=`[4096]/int64C` | is_neox=`False` ; eps=`1e-05` ; head_dim=`128` ; rope_dim=`128` |
| h200 | zimage | q=`[32, 30, 128]/bfloat16C` ; k=`[32, 30, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[32, 128]/float32C` ; positions=`[32]/int64C` | is_neox=`False` ; eps=`1e-05` ; head_dim=`128` ; rope_dim=`128` |
| h200 | zimage | q=`[4128, 30, 128]/bfloat16C` ; k=`[4128, 30, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[4128, 128]/float32C` ; positions=`[4128]/int64C` | is_neox=`False` ; eps=`1e-05` ; head_dim=`128` ; rope_dim=`128` |
| h200 | qwen-edit | q=`[8424, 24, 128]/bfloat16C` ; k=`[8424, 24, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[8424, 128]/float32C` ; positions=`[8424]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |
| h200 | qwen-edit | q=`[195, 24, 128]/bfloat16C` ; k=`[195, 24, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[195, 128]/float32C` ; positions=`[195]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |
| h200 | qwen-edit | q=`[189, 24, 128]/bfloat16C` ; k=`[189, 24, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[189, 128]/float32C` ; positions=`[189]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |
| h200 | flux | q=`[512, 24, 128]/bfloat16C` ; k=`[512, 24, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[4608, 128]/float32C` ; positions=`[512]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |
| h200 | flux | q=`[4096, 24, 128]/bfloat16C` ; k=`[4096, 24, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[4608, 128]/float32C` ; positions=`[4096]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |
| h200 | flux | q=`[4608, 24, 128]/bfloat16C` ; k=`[4608, 24, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[4608, 128]/float32C` ; positions=`[4608]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |
| h200 | flux2 | q=`[512, 48, 128]/bfloat16C` ; k=`[512, 48, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[4608, 128]/float32C` ; positions=`[512]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |
| h200 | flux2 | q=`[4096, 48, 128]/bfloat16C` ; k=`[4096, 48, 128]/bfloat16C` ; q_weight=`[128]/bfloat16C` ; k_weight=`[128]/bfloat16C` ; cos_sin_cache=`[4608, 128]/float32C` ; positions=`[4096]/int64C` | is_neox=`False` ; eps=`1e-06` ; head_dim=`128` ; rope_dim=`128` |

## `rmsnorm_onepass.triton_one_pass_rms_norm`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| b200 | zimage | `[16384, 128]/bfloat16C` ; `[128]/bfloat16C` | `1e-06` |
| b200 | zimage | `[4096, 128]/bfloat16C` ; `[128]/bfloat16C` | `1e-06` |
| h200 | zimage | `[16384, 128]/bfloat16C` ; `[128]/bfloat16C` | `1e-06` |
| h200 | zimage | `[4096, 128]/bfloat16C` ; `[128]/bfloat16C` | `1e-06` |
| h200 | flux2 | `[221184, 128]/bfloat16C` ; `[128]/bfloat16C` | `1e-06` |
| h200 | hunyuanvideo | `[648720, 128]/bfloat16C` ; `[128]/bfloat16C` | `1e-06` |
| h200 | hunyuanvideo | `[1320, 128]/bfloat16C` ; `[128]/bfloat16C` | `1e-06` |
| h200 | hunyuanvideo | `[650040, 128]/bfloat16C` ; `[128]/bfloat16C` | `1e-06` |

## `rotary.apply_rotary_embedding`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| h200 | hunyuanvideo | `[1, 27030, 24, 128]/bfloat16C` ; `[27030, 64]/float32C` ; `[27030, 64]/float32C` | `False` |

## `scale_residual_norm_scale_shift.fused_norm_scale_shift`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| b200 | qwen-edit | `[1, 195, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| b200 | qwen-edit | `[1, 189, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| b200 | wan-ti2v | `[1, 18144, 3072]/bfloat16C` ; `[1, 18144, 3072]/float32C` ; `[1, 18144, 3072]/float32C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | qwen | `[1, 4096, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | qwen | `[1, 19, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | qwen | `[1, 47, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | qwen-edit | `[1, 195, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | qwen-edit | `[1, 189, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | hunyuanvideo | `[1, 27030, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | hunyuanvideo | `[1, 55, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | wan-t2v | `[1, 37800, 5120]/bfloat16C` ; `[1, 1, 5120]/float32C` ; `[1, 1, 5120]/float32C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | wan-ti2v | `[1, 18144, 3072]/bfloat16C` ; `[1, 18144, 3072]/float32C` ; `[1, 18144, 3072]/float32C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | hunyuanvideo | `[1, 27085, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | helios | `[1, 11040, 5120]/bfloat16C` ; `[1, 11040, 5120]/float32C` ; `[1, 11040, 5120]/float32C` | `None` ; `None` ; `layer` ; `1e-06` |

## `scale_residual_norm_scale_shift.fused_scale_residual_norm_scale_shift`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| b200 | qwen-edit | `[1, 195, 3072]/bfloat16C` ; `[1, 195, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| b200 | qwen-edit | `[1, 189, 3072]/bfloat16C` ; `[1, 189, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| b200 | wan-ti2v | `[1, 18144, 3072]/bfloat16C` ; `[1, 18144, 3072]/bfloat16C` ; `[1, 18144, 3072]/float32C` ; `[3072]/float32C` ; `[3072]/float32C` ; `[1]/bfloat16C` ; `[1]/bfloat16C` | `layer` ; `1e-06` |
| b200 | wan-ti2v | `[1, 18144, 3072]/bfloat16C` ; `[1, 18144, 3072]/bfloat16C` ; `[1, 18144, 3072]/float32C` ; `[1, 18144, 3072]/float32C` | `None` ; `None` ; `None` ; `layer` ; `1e-06` |
| h200 | qwen | `[1, 4096, 3072]/bfloat16C` ; `[1, 4096, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | qwen | `[1, 19, 3072]/bfloat16C` ; `[1, 19, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | qwen | `[1, 47, 3072]/bfloat16C` ; `[1, 47, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | qwen-edit | `[1, 195, 3072]/bfloat16C` ; `[1, 195, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | qwen-edit | `[1, 189, 3072]/bfloat16C` ; `[1, 189, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | hunyuanvideo | `[1, 27030, 3072]/bfloat16C` ; `[1, 27030, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | hunyuanvideo | `[1, 55, 3072]/bfloat16C` ; `[1, 55, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` | `None` ; `None` ; `layer` ; `1e-06` |
| h200 | wan-t2v | `[1, 37800, 5120]/bfloat16C` ; `[1, 37800, 5120]/bfloat16C` ; `[1, 1, 5120]/float32C` ; `[5120]/float32C` ; `[5120]/float32C` ; `[1]/bfloat16C` ; `[1]/bfloat16C` | `layer` ; `1e-06` |
| h200 | wan-t2v | `[1, 37800, 5120]/bfloat16C` ; `[1, 37800, 5120]/bfloat16C` ; `[1, 1, 5120]/float32C` ; `[1, 1, 5120]/float32C` | `None` ; `None` ; `None` ; `layer` ; `1e-06` |
| h200 | wan-ti2v | `[1, 18144, 3072]/bfloat16C` ; `[1, 18144, 3072]/bfloat16C` ; `[1, 18144, 3072]/float32C` ; `[3072]/float32C` ; `[3072]/float32C` ; `[1]/bfloat16C` ; `[1]/bfloat16C` | `layer` ; `1e-06` |
| h200 | wan-ti2v | `[1, 18144, 3072]/bfloat16C` ; `[1, 18144, 3072]/bfloat16C` ; `[1, 18144, 3072]/float32C` ; `[1, 18144, 3072]/float32C` | `None` ; `None` ; `None` ; `layer` ; `1e-06` |

## `scale_shift.fuse_layernorm_scale_shift_gate_select01_kernel`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| b200 | qwen-edit | `[1, 8424, 3072]/bfloat16C` ; scale0=`[1, 3072]/bfloat16C` ; shift0=`[1, 3072]/bfloat16C` ; gate0=`[1, 3072]/bfloat16C` ; scale1=`[1, 3072]/bfloat16C` ; shift1=`[1, 3072]/bfloat16C` ; gate1=`[1, 3072]/bfloat16C` ; index=`[1, 8424]/int32C` | weight=`None` ; bias=`None` ; eps=`1e-06` |
| h200 | qwen-edit | `[1, 8424, 3072]/bfloat16C` ; scale0=`[1, 3072]/bfloat16C` ; shift0=`[1, 3072]/bfloat16C` ; gate0=`[1, 3072]/bfloat16C` ; scale1=`[1, 3072]/bfloat16C` ; shift1=`[1, 3072]/bfloat16C` ; gate1=`[1, 3072]/bfloat16C` ; index=`[1, 8424]/int32C` | weight=`None` ; bias=`None` ; eps=`1e-06` |

## `scale_shift.fuse_residual_layernorm_scale_shift_gate_select01_kernel`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| b200 | qwen-edit | `[1, 8424, 3072]/bfloat16C` ; residual=`[1, 8424, 3072]/bfloat16C` ; residual_gate=`[1, 8424, 3072]/bfloat16C` ; scale0=`[1, 3072]/bfloat16C` ; shift0=`[1, 3072]/bfloat16C` ; gate0=`[1, 3072]/bfloat16C` ; scale1=`[1, 3072]/bfloat16C` ; shift1=`[1, 3072]/bfloat16C` ; gate1=`[1, 3072]/bfloat16C` ; index=`[1, 8424]/int32C` | weight=`None` ; bias=`None` ; eps=`1e-06` |
| h200 | qwen-edit | `[1, 8424, 3072]/bfloat16C` ; residual=`[1, 8424, 3072]/bfloat16C` ; residual_gate=`[1, 8424, 3072]/bfloat16C` ; scale0=`[1, 3072]/bfloat16C` ; shift0=`[1, 3072]/bfloat16C` ; gate0=`[1, 3072]/bfloat16C` ; scale1=`[1, 3072]/bfloat16C` ; shift1=`[1, 3072]/bfloat16C` ; gate1=`[1, 3072]/bfloat16C` ; index=`[1, 8424]/int32C` | weight=`None` ; bias=`None` ; eps=`1e-06` |

## `scale_shift.fuse_scale_shift_kernel`

| Arch | Model | Tensor shapes | Other args |
|---|---|---|---|
| b200 | qwen-edit | `[1, 8424, 3072]/bfloat16C` ; `[1, 8424, 3072]/bfloat16C` ; `[1, 8424, 3072]/bfloat16C` | scale_constant=`0` |
| b200 | qwen-edit | `[1, 195, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 195, 3072]/bfloat16C` | scale_constant=`0` |
| b200 | qwen-edit | `[1, 189, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 189, 3072]/bfloat16C` | scale_constant=`0` |
| b200 | wan-ti2v | `[1, 18144, 3072]/bfloat16C` ; `[1, 18144, 3072]/float32NC` ; `[1, 18144, 3072]/bfloat16C` | scale_constant=`0` |
| h200 | qwen | `[1, 4096, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 4096, 3072]/bfloat16C` | scale_constant=`0` |
| h200 | qwen | `[1, 19, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 19, 3072]/bfloat16C` | scale_constant=`0` |
| h200 | qwen | `[1, 47, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 47, 3072]/bfloat16C` | scale_constant=`0` |
| h200 | qwen-edit | `[1, 8424, 3072]/bfloat16C` ; `[1, 8424, 3072]/bfloat16C` ; `[1, 8424, 3072]/bfloat16C` | scale_constant=`0` |
| h200 | qwen-edit | `[1, 195, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 195, 3072]/bfloat16C` | scale_constant=`0` |
| h200 | qwen-edit | `[1, 189, 3072]/bfloat16C` ; `[1, 1, 3072]/bfloat16C` ; `[1, 189, 3072]/bfloat16C` | scale_constant=`0` |
| h200 | hunyuanvideo | `[1, 27030, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 27030, 3072]/bfloat16C` | scale_constant=`0` |
| h200 | hunyuanvideo | `[1, 55, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 55, 3072]/bfloat16C` | scale_constant=`0` |
| h200 | wan-t2v | `[1, 37800, 5120]/bfloat16C` ; `[1, 1, 5120]/float32C` ; `[1, 37800, 5120]/bfloat16C` | scale_constant=`0` |
| h200 | wan-ti2v | `[1, 18144, 3072]/bfloat16C` ; `[1, 18144, 3072]/float32NC` ; `[1, 18144, 3072]/bfloat16C` | scale_constant=`0` |
| h200 | hunyuanvideo | `[1, 27085, 3072]/bfloat16C` ; `[1, 3072]/bfloat16C` ; `[1, 27085, 3072]/bfloat16C` | scale_constant=`0` |

