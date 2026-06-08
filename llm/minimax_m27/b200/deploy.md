# MiniMax-M2.7 — B200 deployment & workflow capture

- **Model:** `MiniMaxAI/MiniMax-M2.7` (FP8, MoE 256 experts / 8 active, ~220 GB)
- **Platform:** B200 ×8, `--tp 8 --ep 8`
- **Docker image:** `lmsysorg/sglang:v0.5.10.post1`
- **Cookbook source:** `sgl-cookbook/docs/autoregressive/MiniMax/MiniMax-M2.7.md`
  (§3 deployment, §5.2 speed benchmark = `random` 1000/1000)

## 1. Serve (TP=8, EP=8, profiler-enabled)

```bash
export SGLANG_TORCH_PROFILER_DIR=$PWD/profile
llm/scripts/serve.sh $PWD/serve.log 2400 -- \
  sglang serve \
    --model-path MiniMaxAI/MiniMax-M2.7 \
    --tp 8 --ep 8 \
    --tool-call-parser minimax-m2 \
    --reasoning-parser minimax-append-think \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --host 0.0.0.0 --port 30000
```

## 2. Benchmark — low / mid / high concurrency (cookbook `random` 1000/1000)

```bash
llm/scripts/bench.sh MiniMaxAI/MiniMax-M2.7 $PWD/bench
# low:1/10, mid:32/300, high:100/500  (cookbook anchors: low c=1, high c=100)
```

## 3. Profile the forward pass (mid concurrency)

```bash
llm/scripts/profile_forward.sh MiniMaxAI/MiniMax-M2.7 $PWD/profile 32 64
```

## 4. Extract the kernel-workflow inventory

```bash
python3 llm/scripts/extract_kernel_workflow.py $PWD/profile \
  --out-md $PWD/docs/kernel_workflow.md \
  --out-csv $PWD/docs/kernel_workflow.csv \
  --label "MiniMax-M2.7 / B200 / mid-conc" --threshold 1.0
```

## 5. Then

- Turn each ≥1% opportunity kernel into a task card under `kernels/`.
- Fill `run_log.md`, commit the folder, push.
- Delete the model weights from the B200 box (HF cache for `MiniMax-M2.7` only).
