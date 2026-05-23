# KernelPilot Prompts

These are end-to-end prompt cards for KernelPilot kernel optimization loops.
They follow the clarity of the
[kernel-design-agents prompt style](https://github.com/mit-han-lab/kernel-design-agents/tree/main/prompts),
but collapse the work into one prompt per task instead of phase-specific
prompts. The FA4 card is seeded from
[KernelPilot issue #1](https://github.com/BBuf/kernel-pilot/issues/1).

| Prompt | Goal |
| --- | --- |
| [B200 int8_scaled_mm](b200-int8-scaled-mm.md) | Optimize SGLang `int8_scaled_mm` on B200 for one focused shape and target at least 2.5x speedup over the SGLang baseline. |
| [B200 FA4 MHA](b200-fa4-mha.md) | Build a standalone BF16 forward-only MHA kernel and beat official FlashAttention-4 by at least 5% geometric-mean TFLOPS across the configured B200 cases. |

Each prompt is intended to be pasted as one complete task. The prompt itself
names the required remote GPU skill and acceptance target.
