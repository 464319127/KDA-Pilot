# b200_tilert_fused_moe (37%)
TileRT FusedMoe — MoE decode (FP4 experts, #2 decode cost 37%, the make-or-break kernel). Target 22µs.
Open-source NOT faster: deep_gemm bf16 grouped = 157µs (~7× slower); needs FP4 + fusion
to reach 22µs. Worth optimizing — see *Open-source baseline comparison* in `../../KERNEL_REGISTRY.md`.
