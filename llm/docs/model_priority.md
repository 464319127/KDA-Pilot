# LLM Model Priority

Priority-1 models (latest flagship from distinct vendors). Each runs on **both**
B200 and H200. Source of deployment commands: sgl-cookbook
(`docs/autoregressive/<Vendor>/<Model>.md`).

| Prio | Model | Vendor | Quant | TP (8-GPU) | Cookbook doc | B200 | H200 |
|---|---|---|---|---|---|---|---|
| 1 | MiniMax-M2.7 | MiniMax | fp8 | tp8 ep8 | `MiniMax/MiniMax-M2.7.md` | ▶ in progress | pending |
| 1 | GLM-5.1 | zai-org | fp8 | tp8 | `GLM/GLM-5.1.md` | pending | pending |
| 1 | Kimi-K2.6 | Moonshotai | int4 | tp8 | `Moonshotai/Kimi-K2.6.md` | pending | pending |
| 1 | DeepSeek-V3.2(-Exp) | DeepSeek | fp8 | tp8 | `DeepSeek/DeepSeek-V3_2.md` | pending | pending |
| 1 | DeepSeek-V4 | DeepSeek | tbd | tbd | live docs (not yet in repo) | pending | pending |

Notes:
- **DeepSeek-V4** is published on the live docs site
  (`https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4`) but is
  not yet in the upstream `sgl-cookbook` repo on any branch. Pull its serve
  command from the live page when we reach it.
- Order today: validate **MiniMax-M2.7** (smallest, ~220 GB) end-to-end on B200
  first, then fan out to the other models × {B200, H200}.
- Run sizing/disk: download one model at a time; delete weights after its folder
  is committed (see `llm_kernel_workflow_rules.md` §5).

## Status legend
`pending` → `downloading` → `serving` → `benchmarked` → `profiled` → `inventory` → `tasks` → `committed` → `cleaned`
