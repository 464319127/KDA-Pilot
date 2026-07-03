# Profile evidence — glm52_bs1__mtp_chain_megakernel

~2.0 ms of the 17.6 ms round is MTP draft refill (extend replay + 6-step
fused chain graph) at 292.6 tok/s baseline (accept 5.16). The chain replays
~40 kernels per step, mostly < 5 µs — launch gaps/tails dominate. Same-
instance A/B history: per-step python replays 283.6 -> fused chain graph
287.4 tok/s; a megakernel continues that direction.

Baseline reproduce: mini-sglang `a26fd6f`, config in `mega_kernels/README.md`;
chain capture code `python/minisgl/engine/graph.py::_capture_mtp_chain`.
