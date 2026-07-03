# B300 access runbook (verda fleet, radix)

The only B300 machines we have are the Verda 8x B300 SXM6 siblings
`verda-b300-fin-03-{2,3,4}` behind one bastion. Leases come from the radix
CLI; the `bbuf` account has **0 credits**, which shapes everything below.

## 1. Get / restore a lease

```bash
export PATH="$HOME/.local/bin:$PATH"
export RADIX_API=https://nodes.sglang.io
radix machines mine                      # trust this over any remembered expiry
radix assign verda-b300-fin-03-3         # free; default lease 4h
# CLI wants a TTY for its picker — in headless shells wrap it:
script -q /dev/null radix assign verda-b300-fin-03-3
```

- `radix extend` does NOT work (needs credits). Leases just expire — SSH then
  fails with `Permission denied (publickey)`. **Re-assign is free** and
  restores access; `/data/bbuf` survives the lapse (not scrubbed).
- If assign says "you already have an assignment — release it first":
  `radix release verda-b300-fin-03-3 && radix assign verda-b300-fin-03-3`.
- If the FIRST hop (`ubuntu@95.133.252.66`) refuses your key after a
  release/assign cycle, the bastion key propagation lapsed — retry after a
  few minutes; if it persists, ping ops to re-add the key (radix prints this
  hint itself).
- Long jobs: re-assign proactively before the 4h mark or the run dies with
  the SSH session.

## 2. SSH (always via the bastion)

```bash
ssh -i ~/.ssh/id_ed25519 -J ubuntu@95.133.252.66 bbuf@light-face-hides-fin-03-3
# siblings: light-face-hides-fin-03-2 / -4 — pick whichever has free GPUs
```

- The verda nodes cannot SSH each other or reach GitHub with push creds:
  **sync files by base64 through the laptop**:
  `B64=$(base64 < file); ssh ... "echo '$B64' | base64 -d > /remote/path"`.
- Background jobs: launch with `docker exec -d` (plain `&` over ssh dies
  with the connection).

## 3. On the node

```bash
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader  # check contention FIRST
docker exec -it sglang_new bash        # our container (recent sglang:dev; bbuf is in the docker group, no sudo needed)
```

- Workspace `/data/bbuf`; mini-sglang checkout `/data/bbuf/repos/mini-sglang`
  (run with `PYTHONPATH=/data/bbuf/repos/mini-sglang/python`); GLM-5.2-FP8
  weights `/data/bbuf/glm52_real` (~704 GB).
- **Contention:** a tenant container `glm_pd` intermittently grabs ALL 8 GPUs
  (~253 GB each). Do not kill it. If the node is grabbed, try the -2/-4
  siblings (weights may need `hf download zai-org/GLM-5.2-FP8 --local-dir ...`,
  ~30 min at ~425 MB/s, public repo).
- Never start a TP=8 server while another process holds one GPU — the
  memory-imbalance check crashes the load.

## 4. GPU budget per task type

- **Single-GPU kernel tasks** (`fused_moe_decode_fp8`, `skinny_gemm_bf16_tc`,
  `mla_decode_bs_consistent`, `silu_mul_quant_fp8_bitwise`): develop and
  benchmark on ONE idle GPU — `CUDA_VISIBLE_DEVICES=7` inside `sglang_new`
  is the established pattern and coexists with a running TP=8 server on
  GPUs 0-7 only if that server was started first and you only take the
  leftover memory (safer: pick a moment the server is down).
- **8-GPU tasks** (`oneshot_allreduce_bf16`, `mtp_chain_megakernel`): need
  the whole node idle (kill our own server first:
  `docker exec sglang_new bash -c 'pkill -9 -f "minisg[l]"; pkill -9 -f "spawn_mai[n]"'`).
- e2e validation (tier-B accept A/B) needs the full server; launch command is
  in each task's `docs/profile_evidence.md`.
