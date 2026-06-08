# KDA-Pilot

KDA-Pilot is a lightweight prompt repository for standalone GPU-kernel
optimization tasks, organized into two parallel campaigns:

```text
diffusion/    SGLang diffusion-operator kernel tasks (CUDA / Triton / CuTe-DSL),
              each a self-contained optimization task with baseline, candidate,
              benchmark, and results. See diffusion/README.md.

llm/          SGLang autoregressive-model kernel-workflow campaign. For each
              priority LLM, serve it on B200/H200 via the sgl-cookbook command,
              benchmark at low/mid/high concurrency, profile the forward pass,
              and record every >=1% kernel (excluding attention/cuDNN) as a
              kernel-optimization task. See llm/README.md.

external/     Shared knowledge submodules used by both campaigns:
                KernelWiki/         Blackwell/Hopper kernel design references
                ncu-report-skill/   Nsight Compute profiling/report helper
```

Both subtrees share one git history and the `external/` submodules. Run the
diffusion launchers from the repo root with the `diffusion/` prefix, e.g.
`diffusion/scripts/launch_kernels/k03_*.sh`.

External knowledge submodules are optional supporting material:

```bash
git submodule update --init --recursive
```
