# MiniMax-M2.7 / B200 — run log

Provenance for the workflow-capture run. Fill during execution; benchmark
numbers are valid only if the selected GPUs were idle before AND after.

| Field | Value |
|---|---|
| Status | pending |
| Host | ion-b200 (to confirm) |
| GPUs | tp=8 ep=8 — gpu ids: _to fill_ |
| Idle before | _to fill_ |
| Idle after | _to fill_ |
| Docker image | lmsysorg/sglang:v0.5.10.post1 |
| SGLang version/commit | _to fill_ |
| Model | MiniMaxAI/MiniMax-M2.7 (fp8) |
| Serve cmd | see deploy.md §1 |
| Dataset | random, in=1000 out=1000 |
| Concurrency levels | low c=1 (np 10), mid c=32 (np 300), high c=100 (np 500) |
| Trace | profile/_to fill_ |
| KDA-Pilot commit | _to fill_ |

## Benchmark summary (filled from bench/ logs)

| Level | Conc | Req/s | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---|---|---|---|---|---|
| low | 1 | | | | | |
| mid | 32 | | | | | |
| high | 100 | | | | | |

## Notes / deviations

- _record any deviation from the cookbook command, OOM, GPU contention, etc._
