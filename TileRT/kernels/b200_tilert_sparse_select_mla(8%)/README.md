# b200_tilert_sparse_select_mla (8%)

TileRT `SparseSelectMlaDsv32` — GPU0 self-MLA over selected KV.
Open-source NOT faster: same MLA-decode structure as PureMla (~12µs isolated); flashinfer
MLA = 25.5µs isolated (~2× slower). Worth optimizing — see *Open-source baseline
comparison* in `../../KERNEL_REGISTRY.md`.
