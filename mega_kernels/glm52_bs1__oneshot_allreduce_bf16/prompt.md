# KDA Prompt: glm52_bs1__oneshot_allreduce_bf16

Target GPU: 8x NVIDIA B300 SXM6, NVLink with **NVLS (multimem)**. Build a
one-shot all-reduce for tiny bf16 tensors that beats NCCL NVLS —
**~11% of the round** (~1.9 ms) on the GLM-5.2 bs=1 deployment.

## Problem (FIXED)

All-reduce across TP=8 ranks, called ~156x per decode round (2 per layer:
after attention o_proj and after MoE routed+shared), always inside CUDA
graphs:

```
tensor: bf16 [M, 6144], M ∈ {1..8} (verify M=8 dominates; chain steps M=1)
world : 8 ranks, one process per GPU (mini-sglang scheduler), NVLink NVLS
```

## Baselines (measured in-graph on this system)

| impl | latency @ [8,6144] | notes |
|---|---:|---|
| NCCL NVLS (in-graph, warm) | 12.1 µs | current production |
| vLLM-style one-shot (IPC LD/ST, no multimem) | 20.4-32 µs | prior art, LOST — do not repeat; `MINISGL_CUSTOM_AR=0` default |
| TileRT-class reference | ~5-6 µs | closed implementation, same class of hardware |

**Success: ≤ 8 µs at [8, 6144] and ≤ 6 µs at [1, 6144]**, bitwise-stable
ordering (deterministic reduction order across replays; NCCL NVLS is NOT
deterministic and that nondeterminism is measurable as a ±8 tok/s e2e band —
a deterministic winner is worth extra).

## Approach requirements

- Use NVLS multimem: `multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2` +
  `multimem.st` over a multicast-bound symmetric buffer (cuMulticastCreate /
  cuMemMap path, or hijack NCCL's NVLS window). The LD/ST-only design is
  proven dead on this fleet (see baselines).
- Must be CUDA-graph capturable (no host sync inside), PDL-friendly
  (`programmaticStreamSerialization` attr optional), one block preferred
  (48 KB payload → a single 256-thread block streams it in ~2 µs).
- Barrier: system-scope release/acquire flags in multicast space, one flag
  per rank per graph-node occurrence (rotate a small flag pool; graphs replay
  the same nodes so the pool must be replay-safe).
- Numerics: fp32 accumulate before bf16 store (matches NCCL); deployment is
  tier B but comm reordering sits at the ~1e-6 NVLS band which is measured
  safe (README rule 3) — still run the accept A/B once.

## Harness notes

Benchmark with 8 processes (torchrun or mini-sglang's spawn), CUDA-graph
timed, message resident in the multicast buffer (production writes the
o_proj/MoE output directly into it — design the API as
`ar.buffer(M) -> tensor` + `ar.reduce()` so the copy disappears).
Correctness oracle: torch.distributed all_reduce fp32 reference, plus a
determinism check (two replays, bitwise-equal outputs).

## Hardware access (B300)

Full runbook: `../docs/b300_access.md`. **This task needs the WHOLE 8-GPU node**
(NVLink NVLS + 8 processes):

```bash
export PATH="$HOME/.local/bin:$PATH"; export RADIX_API=https://nodes.sglang.io
radix assign verda-b300-fin-03-3       # free re-assign on every lease lapse (4h, no extend possible)
ssh -i ~/.ssh/id_ed25519 -J ubuntu@95.133.252.66 bbuf@light-face-hides-fin-03-3
nvidia-smi                              # ALL 8 GPUs must be free (glm_pd tenant grabs them at times -> use fin-03-2/-4)
docker exec sglang_new bash -c 'pkill -9 -f "minisg[l]"; pkill -9 -f "spawn_mai[n]"'  # stop our server first
docker exec -it sglang_new bash        # torchrun --nproc_per_node 8 bench/...
```

Long multi-process benches die with the SSH session — always `docker exec -d`
+ log file, and re-assign the lease before the 4h mark. File sync is
base64-over-ssh (no push creds on the node).

Follow `../../llm/docs/llm_kernel_optimization_rules.md` and
`../../llm/docs/llm_correctness_contract.md`. Prior-art autopsy of the failed
LD/ST version: mini-sglang `python/minisgl/distributed/impl.py`
(`MINISGL_CUSTOM_AR` plugin) — copy into `docs/prior_art/`.
