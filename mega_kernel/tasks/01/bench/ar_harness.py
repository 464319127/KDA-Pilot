"""Authoritative 8-GPU A/B harness for the MNNVL fused allreduce+add+rmsnorm task.

Implements config.toml's benchmark mode: "8x per-device 50-round CUDA graphs,
concurrent replay, wall/round" (pattern: prior/ar_bench.py), on the shared
single-process NVLS workspace (bench/sp_nvls_workspace.py), for the frozen
workloads in bench/workloads.json (T in {1,6}, H=6144, bf16, world=8,
eps=1e-5).

Preserved template policies (see docs/benchmark_method.md, deviation D3):
frozen workloads, fresh inputs per trial, randomized A/B order per trial,
output poisoning, median/mean/std/min/p10/p90, speedup = baseline_median /
candidate_median, no-silent-skip, JSONL records with full provenance
(all-8-GPU state before/after).

Modes:
  correctness  eager rounds; bitwise A/B (port vs original) + fp32 oracle at
               contract tolerance on every rank/row
  bench        graph-replay wall/round timing per implementation
  stability    N graph replays with per-round in-graph bitwise mismatch
               accumulation against a precomputed reference (race detector)
  noise        the bench protocol with the SAME impl on both sides -> the
               measured noise floor for "beyond noise" judgments
  selftest     comparator self-tests (must-fail negative cases)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Dict, List

import torch

TASK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_ROOT)

from bench import correctness as cc  # noqa: E402
from bench.sp_nvls_workspace import SpNvlsWorkspace  # noqa: E402

EPS = 1e-5
HIDDEN = 6144
WORLD = 8


# ----------------------------------------------------------------------
# implementations
# ----------------------------------------------------------------------


def load_impl(name: str):
    if name == "fi":
        from baseline import fi_original

        return fi_original
    if name == "jit":
        from solution import jit_port

        return jit_port
    if name == "jits":  # diagnostic: the serving-built .so (sglang load_jit)
        from solution import jit_serving_build

        return jit_serving_build
    if name == "opt":  # P1 candidate: bs=1 constant-specialized entry
        from solution import jit_port_opt

        return jit_port_opt
    if name == "optba":  # experimental block-arrival variant (route eval only)
        from solution import jit_port_opt_ba

        return jit_port_opt_ba
    raise ValueError(f"unknown impl {name!r} (expected fi|jit|jits|opt|optba)")


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------


class RowIO:
    """Per-trial tensors for one workload row; identical values feed every impl."""

    def __init__(self, T: int, seed: int):
        gen = torch.Generator().manual_seed(seed)
        gamma_cpu = torch.randn(HIDDEN, generator=gen, dtype=torch.float32).to(torch.bfloat16)
        residual_cpu = torch.randn(T, HIDDEN, generator=gen, dtype=torch.float32).to(
            torch.bfloat16
        )
        xs_cpu = [
            torch.randn(T, HIDDEN, generator=gen, dtype=torch.float32).to(torch.bfloat16)
            for _ in range(WORLD)
        ]
        self.T = T
        self.xs, self.residuals, self.gammas = [], [], []
        for i in range(WORLD):
            dev = f"cuda:{i}"
            torch.cuda.set_device(i)
            self.xs.append(xs_cpu[i].to(dev))
            self.residuals.append(residual_cpu.to(dev))  # replicated bytes
            self.gammas.append(gamma_cpu.to(dev))

    def make_outputs(self):
        outs, resouts = [], []
        for i in range(WORLD):
            torch.cuda.set_device(i)
            outs.append(
                cc.poison_(torch.empty(self.T, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{i}"))
            )
            resouts.append(
                cc.poison_(torch.empty(self.T, HIDDEN, dtype=torch.bfloat16, device=f"cuda:{i}"))
            )
        return outs, resouts

    def oracle(self) -> cc.OracleRefs:
        return cc.fp32_oracle(
            [x.to("cuda:0") for x in self.xs],
            self.residuals[0].to("cuda:0"),
            self.gammas[0].to("cuda:0"),
            EPS,
        )


# ----------------------------------------------------------------------
# launch helpers
# ----------------------------------------------------------------------


def eager_round(impl, io: RowIO, ws, outs, resouts, pdl: bool):
    for i in range(WORLD):
        torch.cuda.set_device(i)
        impl.launch(io.xs[i], outs[i], io.residuals[i], resouts[i], io.gammas[i],
                    ws.ranks[i], EPS, pdl)
    for i in range(WORLD):
        torch.cuda.synchronize(i)


def capture_graphs(impl, io: RowIO, ws, outs, resouts, rounds: int, pdl: bool,
                   streams, mismatch=None, refs=None):
    """Capture one graph per device with `rounds` back-to-back launches.

    When `mismatch`/`refs` are given, each round is followed by an in-graph
    bitwise mismatch accumulation (stability instrumentation).
    """
    graphs = []
    for i in range(WORLD):
        torch.cuda.set_device(i)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(streams[i]):
            with torch.cuda.graph(g, stream=streams[i]):
                for _ in range(rounds):
                    impl.launch(io.xs[i], outs[i], io.residuals[i], resouts[i],
                                io.gammas[i], ws.ranks[i], EPS, pdl)
                    if mismatch is not None:
                        ref_out, ref_res = refs[i]
                        mismatch[i].add_(
                            outs[i].view(torch.int16).ne(ref_out).sum()
                            + resouts[i].view(torch.int16).ne(ref_res).sum()
                        )
        graphs.append(g)
    return graphs


def replay_all(graphs):
    for i in range(WORLD):
        torch.cuda.set_device(i)
        graphs[i].replay()


def sync_all():
    for i in range(WORLD):
        torch.cuda.synchronize(i)


def timed_replays(graphs, reps: int, rounds: int) -> float:
    """ar_bench convention: reps x (concurrent replay of all 8, then sync all).

    Returns wall microseconds per round.
    """
    t0 = time.perf_counter()
    for _ in range(reps):
        replay_all(graphs)
        sync_all()
    return (time.perf_counter() - t0) / (reps * rounds) * 1e6


# ----------------------------------------------------------------------
# provenance
# ----------------------------------------------------------------------


def gpu_state() -> List[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        return out.splitlines()
    except Exception as e:  # provenance must never crash the run
        return [f"nvidia-smi failed: {e}"]


def _sha16(path: str) -> str:
    import hashlib

    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    except Exception:
        return "unavailable"


def provenance(args) -> Dict:
    import flashinfer

    try:
        import tvm_ffi

        tvm_ver = getattr(tvm_ffi, "__version__", "unknown")
    except Exception:
        tvm_ver = "unknown"
    try:
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True,
                              text=True, timeout=20).stdout.strip().splitlines()[-1]
    except Exception:
        nvcc = "unavailable"
    try:
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    except Exception:
        gpus = ["unavailable"]
    csrc = os.path.join(TASK_ROOT, "solution", "mnnvl_ar_fused", "csrc")
    build_info = None
    try:
        from solution import jit_port

        build_info = jit_port.build_info()
    except Exception:
        pass
    return {
        "task": "mega_kernel/tasks/01 (mnnvl_ar_jit_bs1)",
        "host": os.uname().nodename,
        "command": " ".join(sys.argv),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "tvm_ffi": tvm_ver,
        "nvcc": nvcc,
        "python": sys.version.split()[0],
        "gpus": gpus,
        "world": WORLD,
        "hidden": HIDDEN,
        "eps": EPS,
        "source_hashes": {
            "baseline_header": _sha16(os.path.join(TASK_ROOT, "baseline",
                                                   "trtllm_mnnvl_allreduce.cuh")),
            "port_cuh": _sha16(os.path.join(csrc, "mnnvl_ar_fused.cuh")),
            "port_binding_cu": _sha16(os.path.join(csrc, "mnnvl_ar_fused.cu")),
            "port_compat": _sha16(os.path.join(csrc, "mnnvl_ar_fused_compat.cuh")),
            "port_opt_cuh": _sha16(os.path.join(csrc, "mnnvl_ar_fused_opt.cuh")),
        },
        "candidate_build": build_info,
        "settings": {
            "rounds": args.rounds, "reps": args.reps, "trials": args.trials,
            "warmup_replays": args.warmup_replays, "pdl": args.pdl,
            "seed": args.seed,
        },
    }


def stats(samples: List[float]) -> Dict:
    ss = sorted(samples)
    n = len(ss)
    return {
        "n": n,
        "median": statistics.median(ss),
        "mean": statistics.fmean(ss),
        "std": statistics.stdev(ss) if n > 1 else 0.0,
        "min": ss[0],
        "p10": ss[max(0, int(0.10 * (n - 1)))],
        "p90": ss[min(n - 1, int(0.90 * (n - 1)))],
    }


def emit(record: Dict, out_path: str):
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ----------------------------------------------------------------------
# modes
# ----------------------------------------------------------------------


def load_rows(args) -> List[Dict]:
    with open(os.path.join(TASK_ROOT, "bench", "workloads.json")) as f:
        rows = json.load(f)["workloads"]
    if args.tokens != "all":
        rows = [r for r in rows if r["scalars"]["num_tokens"] == int(args.tokens)]
    if not rows:
        raise SystemExit("no workload rows selected — refusing to run (no silent skips)")
    return rows


def mode_correctness(args, ws) -> bool:
    impls = args.impls.split(",")
    all_ok = True
    for row in load_rows(args):
        T = row["scalars"]["num_tokens"]
        io = RowIO(T, args.seed)
        oracle = io.oracle()
        per_impl = {}
        for name in impls:
            impl = load_impl(name)
            ws.reset()
            outs, resouts = io.make_outputs()
            for _ in range(args.correctness_rounds):
                eager_round(impl, io, ws, outs, resouts, bool(args.pdl))
            per_impl[name] = (outs, resouts)
            # structural + oracle at contract tolerance on every rank
            for i in range(WORLD):
                msg = cc.structural_check(f"{name}/rank{i}/out", outs[i], outs[0])
                ok_o, ex_o = cc.contract_allclose(outs[i], oracle.out.to(outs[i].device))
                ok_r, ex_r = cc.contract_allclose(
                    resouts[i], oracle.residual_out.to(resouts[i].device))
                if msg or not ok_o or not ok_r:
                    all_ok = False
                    print(f"[correctness] T={T} {name} rank{i} FAIL "
                          f"struct={msg} oracle_out_excess={ex_o:.3e} "
                          f"oracle_res_excess={ex_r:.3e}")
            print(f"[correctness] T={T} {name}: oracle max-rel(out)="
                  f"{max(cc.rel_err(per_impl[name][0][i], oracle.out.to(f'cuda:{i}')) for i in range(WORLD)):.3e} "
                  f"(bf16 ulp floor ~3.9e-3), contract-tolerance PASS={all_ok}")
        if len(impls) == 2:
            a, b = impls
            for i in range(WORLD):
                for kind, ta, tb in (("out", per_impl[a][0][i], per_impl[b][0][i]),
                                     ("residual_out", per_impl[a][1][i], per_impl[b][1][i])):
                    eq, count, first = cc.bitwise_equal(ta, tb)
                    if not eq:
                        all_ok = False
                        print(f"[correctness] T={T} BITWISE {a} vs {b} rank{i} {kind}: "
                              f"{count} mismatches, first at flat index {first}")
            if all_ok:
                print(f"[correctness] T={T}: {a} vs {b} bf16 BIT-EXACT on all "
                      f"{WORLD} ranks (out + residual_out)")
    return all_ok


def mode_bench(args, ws, impl_names=None) -> bool:
    impl_names = impl_names or args.impls.split(",")
    import random

    rng = random.Random(args.seed)
    before = gpu_state()
    prov = provenance(args)
    ok = True
    headline_speedups = []
    for row in load_rows(args):
        T = row["scalars"]["num_tokens"]
        # Tie timing to correctness: when comparing two impls, refuse to time
        # unless they are bit-exact on this row right now (same build).
        pretiming_bitexact = None
        if len(impl_names) == 2:
            io0 = RowIO(T, args.seed)
            per = {}
            for name in impl_names:
                impl = load_impl(name)
                ws.reset()
                o, r = io0.make_outputs()
                eager_round(impl, io0, ws, o, r, bool(args.pdl))
                per[name] = (o, r)
            a, b = impl_names
            pretiming_bitexact = all(
                cc.bitwise_equal(per[a][0][i], per[b][0][i])[0]
                and cc.bitwise_equal(per[a][1][i], per[b][1][i])[0]
                for i in range(WORLD)
            )
            if not pretiming_bitexact:
                print(f"[bench] T={T}: PRE-TIMING BIT-EXACT CHECK FAILED — "
                      f"refusing to time (no silent wrong measurements)")
                ok = False
                continue
        samples: Dict[str, List[float]] = {n: [] for n in impl_names}
        streams = [torch.cuda.Stream(device=i) for i in range(WORLD)]
        for trial in range(args.trials):
            io = RowIO(T, args.seed + trial * 1_000_003 + T)
            order = list(impl_names)
            rng.shuffle(order)
            for name in order:
                impl = load_impl(name)
                ws.reset()
                outs, resouts = io.make_outputs()
                # eager warm (JIT/op registration + allocator), then clean state
                eager_round(impl, io, ws, outs, resouts, bool(args.pdl))
                ws.reset()
                graphs = capture_graphs(impl, io, ws, outs, resouts, args.rounds,
                                        bool(args.pdl), streams)
                for _ in range(args.warmup_replays):
                    replay_all(graphs)
                    sync_all()
                us = timed_replays(graphs, args.reps, args.rounds)
                samples[name].append(us)
                del graphs
        row_rec = {"mode": "bench", "row": row["id"], "T": T,
                   "provenance": prov, "gpu_before": before,
                   "pretiming_bitexact": pretiming_bitexact,
                   "raw_samples_us": samples,
                   "results": {}}
        base = impl_names[0]
        for name in impl_names:
            row_rec["results"][name] = stats(samples[name])
        if len(impl_names) == 2:
            cand = impl_names[1]
            sp = row_rec["results"][base]["median"] / row_rec["results"][cand]["median"]
            row_rec["speedup_median"] = sp
            if row.get("headline", row.get("production", True)):
                headline_speedups.append(sp)
            print(f"[bench] T={T}: {base} {row_rec['results'][base]['median']:.3f}us "
                  f"vs {cand} {row_rec['results'][cand]['median']:.3f}us "
                  f"speedup={sp:.4f} (pretiming bit-exact: {pretiming_bitexact})")
        else:
            print(f"[bench] T={T}: {base} median={row_rec['results'][base]['median']:.3f}us "
                  f"mean={row_rec['results'][base]['mean']:.3f} "
                  f"std={row_rec['results'][base]['std']:.3f} "
                  f"min={row_rec['results'][base]['min']:.3f} (n={args.trials})")
        row_rec["gpu_after"] = gpu_state()
        emit(row_rec, args.out)
    if headline_speedups:
        import math

        geo = math.exp(sum(math.log(s) for s in headline_speedups) / len(headline_speedups))
        arith = sum(headline_speedups) / len(headline_speedups)
        emit({"mode": "bench_headline", "impls": impl_names,
              "geomean_speedup": geo, "arith_mean_speedup": arith,
              "rows": len(headline_speedups), "provenance": prov}, args.out)
        print(f"[bench] HEADLINE geomean speedup over {len(headline_speedups)} "
              f"production rows: {geo:.4f} (arith {arith:.4f})")
    return ok


def mode_noise(args, ws) -> bool:
    """A/B protocol with the same impl on both slots -> noise floor."""
    name = args.impls.split(",")[0]
    import random

    rng = random.Random(args.seed)
    before = gpu_state()
    prov = provenance(args)
    for row in load_rows(args):
        T = row["scalars"]["num_tokens"]
        streams = [torch.cuda.Stream(device=i) for i in range(WORLD)]
        sa, sb = [], []
        for trial in range(args.trials):
            io = RowIO(T, args.seed + trial * 1_000_003 + T)
            impl = load_impl(name)
            pair = [("A", sa), ("B", sb)]
            rng.shuffle(pair)
            for _tag, bucket in pair:
                ws.reset()
                outs, resouts = io.make_outputs()
                eager_round(impl, io, ws, outs, resouts, bool(args.pdl))
                ws.reset()
                graphs = capture_graphs(impl, io, ws, outs, resouts, args.rounds,
                                        bool(args.pdl), streams)
                for _ in range(args.warmup_replays):
                    replay_all(graphs)
                    sync_all()
                bucket.append(timed_replays(graphs, args.reps, args.rounds))
                del graphs
        ra, rb = stats(sa), stats(sb)
        pseudo = ra["median"] / rb["median"]
        spread = abs(1.0 - pseudo)
        rec = {"mode": "noise", "row": row["id"], "T": T, "impl": name,
               "provenance": prov, "gpu_before": before, "gpu_after": gpu_state(),
               "A": ra, "B": rb, "pseudo_speedup": pseudo, "noise_spread": spread}
        emit(rec, args.out)
        print(f"[noise] T={T} {name}: A={ra['median']:.3f}us B={rb['median']:.3f}us "
              f"pseudo-speedup={pseudo:.4f} (|1-x|={spread:.4f}) "
              f"stdA={ra['std']:.3f} stdB={rb['std']:.3f}")
    return True


def mode_stability(args, ws) -> bool:
    """Race detector with round-distinguishable data (review-hardened).

    Design (addresses the two blind spots of naive final-output checking):
    - THREE rotating input banks: each in-graph round first copies bank
      (r mod 3) into the live input tensors, so a stale read from a previous
      Lamport epoch carries DIFFERENT values and cannot compare equal. Three
      banks cover staleness of 1 or 2 epochs (the buffer ring depth).
    - In-graph output poisoning (NaN fill) before every launch: a silently
      skipped/no-op round leaves NaN and cannot equal any reference.
    - Per-bank references computed from clean serialized eager rounds of the
      SAME implementation (bit-exact self-consistency is the property under
      test; cross-impl equality is mode_correctness's job).
    Additionally runs a plain uninstrumented graph (maximum cross-round
    overlap, production-shaped) with a final-output check after every replay.
    """
    name = args.impls.split(",")[0]
    impl = load_impl(name)
    ok = True
    NBANK = 3
    for row in load_rows(args):
        T = row["scalars"]["num_tokens"]
        streams = [torch.cuda.Stream(device=i) for i in range(WORLD)]
        banks = [RowIO(T, args.seed + 7_777_777 * b) for b in range(NBANK)]
        io = RowIO(T, args.seed)  # live tensors the graph reads
        outs, resouts = io.make_outputs()

        # Banks differ in xs ONLY (residual/gamma stay io's values throughout,
        # matching production where weights are constant across calls); the
        # raced data is the x broadcast through the Lamport buffers, so
        # distinguishable xs is what detects stale-epoch reads.
        refs = []  # refs[b][rank] = (out_int16, resout_int16)
        for b in range(NBANK):
            for i in range(WORLD):
                io.xs[i].copy_(banks[b].xs[i])
            ws.reset()
            for i in range(WORLD):
                cc.poison_(outs[i]); cc.poison_(resouts[i])
            eager_round(impl, io, ws, outs, resouts, bool(args.pdl))
            refs.append([(outs[i].view(torch.int16).clone(),
                          resouts[i].view(torch.int16).clone()) for i in range(WORLD)])

        mismatch = [torch.zeros((), dtype=torch.int64, device=f"cuda:{i}")
                    for i in range(WORLD)]

        # --- instrumented graph: copy bank, poison, launch, compare ---
        ws.reset()
        graphs = []
        for i in range(WORLD):
            torch.cuda.set_device(i)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=streams[i]):
                for r in range(args.rounds):
                    b = r % NBANK
                    io.xs[i].copy_(banks[b].xs[i])
                    outs[i].fill_(float("nan"))
                    resouts[i].fill_(float("nan"))
                    impl.launch(io.xs[i], outs[i], io.residuals[i], resouts[i],
                                io.gammas[i], ws.ranks[i], EPS, bool(args.pdl))
                    ref_out, ref_res = refs[b][i]
                    mismatch[i].add_(
                        outs[i].view(torch.int16).ne(ref_out).sum()
                        + resouts[i].view(torch.int16).ne(ref_res).sum()
                    )
            graphs.append(g)
        t0 = time.time()
        for rep in range(args.stability_replays):
            replay_all(graphs)
            sync_all()
            if (rep + 1) % 200 == 0:
                total = sum(int(m.item()) for m in mismatch)
                print(f"[stability] T={T} {name}: {rep + 1}/{args.stability_replays} "
                      f"instrumented replays, cumulative mismatches={total}", flush=True)
        total_instr = sum(int(m.item()) for m in mismatch)
        checked_instr = args.stability_replays * args.rounds
        del graphs

        # --- plain graph (production-shaped, max overlap), final check per replay
        for i in range(WORLD):
            io.xs[i].copy_(banks[0].xs[i])
        ws.reset()
        for i in range(WORLD):
            cc.poison_(outs[i]); cc.poison_(resouts[i])
        plain = capture_graphs(impl, io, ws, outs, resouts, args.rounds,
                               bool(args.pdl), streams)
        plain_bad = 0
        for rep in range(args.stability_replays):
            replay_all(plain)
            sync_all()
            for i in range(WORLD):
                if not torch.equal(outs[i].view(torch.int16), refs[0][i][0]) or \
                        not torch.equal(resouts[i].view(torch.int16), refs[0][i][1]):
                    plain_bad += 1
        del plain

        rec = {"mode": "stability", "row": row["id"], "T": T, "impl": name,
               "replays": args.stability_replays, "rounds_per_replay": args.rounds,
               "instrumented_rounds_checked": checked_instr,
               "instrumented_mismatched_elements": total_instr,
               "plain_final_check_failures": plain_bad,
               "input_banks": NBANK,
               "elapsed_s": time.time() - t0, "provenance": provenance(args)}
        emit(rec, args.out)
        verdict = "STABLE" if (total_instr == 0 and plain_bad == 0) else "UNSTABLE"
        print(f"[stability] T={T} {name}: instrumented {checked_instr} rounds "
              f"(3 rotating banks, in-graph poison) mismatches={total_instr}; "
              f"plain {args.stability_replays} replays final-check failures="
              f"{plain_bad} -> {verdict}")
        ok = ok and total_instr == 0 and plain_bad == 0
    return ok


def _feed_solo_slots(ws, rank: int, io: RowIO, T: int) -> None:
    """Pre-feed rank's Lamport buffer 0 with all 8 sanitized shards.

    Reproduces the kernel's broadcast content (including the -0.0 -> +0.0
    payload sanitize), so a SOLO launch of this rank's kernel finds every
    slot valid and never spins on live peers. This makes single-kernel NCU
    profiling safe (kernel replay restores this device's memory between
    passes) and yields the compute-floor latency (harness wall/round minus
    this is the communication-wait share).
    """
    neg_zero = torch.tensor(-32768, dtype=torch.int16, device=f"cuda:{rank}")
    for rr in range(WORLD):
        x = io.xs[rr].to(f"cuda:{rank}")
        xi = x.view(torch.int16)
        x_san = torch.where(xi == neg_zero, torch.zeros_like(xi), xi).view(torch.bfloat16)
        for t in range(T):
            byte_off = (t * HIDDEN * WORLD + rr * HIDDEN) * 2  # Lamport buffer 0
            ws.memcpy_into_uc(rank, byte_off, x_san[t].contiguous())
    torch.cuda.synchronize(rank)


def mode_pdlprobe(args, ws) -> bool:
    """PDL-with-predecessor evaluation (P1 idea-pool route closure).

    Isolated back-to-back AR rounds cannot show PDL benefit (nothing to
    overlap with). This mode captures per-device graphs where every round is
    [predecessor kernel -> AR launch]: the predecessor copies the input from
    a source buffer (standing in for the producer GEMM writing x), so with
    launch_with_pdl=1 the AR kernel's launch/entry can overlap the
    predecessor's tail exactly as in the serving graph. Compares pdl=0 vs
    pdl=1 per implementation per row (pair wall/round includes the
    predecessor both ways; the delta isolates PDL), with a post-replay
    bitwise correctness check against an eager reference.
    """
    impl_names = args.impls.split(",")
    prov = provenance(args)
    all_ok = True
    for row in load_rows(args):
        T = row["scalars"]["num_tokens"]
        streams = [torch.cuda.Stream(device=i) for i in range(WORLD)]
        io = RowIO(T, args.seed)
        x_srcs = [io.xs[i].clone() for i in range(WORLD)]
        rec = {"mode": "pdlprobe", "row": row["id"], "T": T,
               "provenance": prov, "results": {}}
        for name in impl_names:
            impl = load_impl(name)
            # eager reference (predecessor applied once)
            ws.reset()
            outs, resouts = io.make_outputs()
            for i in range(WORLD):
                io.xs[i].copy_(x_srcs[i])
            eager_round(impl, io, ws, outs, resouts, False)
            refs = [(outs[i].view(torch.int16).clone(),
                     resouts[i].view(torch.int16).clone()) for i in range(WORLD)]
            for pdl in (0, 1):
                samples = []
                for trial in range(args.trials):
                    ws.reset()
                    for i in range(WORLD):
                        cc.poison_(outs[i]); cc.poison_(resouts[i])
                    graphs = []
                    for i in range(WORLD):
                        torch.cuda.set_device(i)
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, stream=streams[i]):
                            for _ in range(args.rounds):
                                io.xs[i].copy_(x_srcs[i])  # predecessor kernel
                                impl.launch(io.xs[i], outs[i], io.residuals[i],
                                            resouts[i], io.gammas[i], ws.ranks[i],
                                            EPS, bool(pdl))
                        graphs.append(g)
                    for _ in range(args.warmup_replays):
                        replay_all(graphs)
                        sync_all()
                    samples.append(timed_replays(graphs, args.reps, args.rounds))
                    del graphs
                st = stats(samples)
                ok_bits = all(
                    torch.equal(outs[i].view(torch.int16), refs[i][0])
                    and torch.equal(resouts[i].view(torch.int16), refs[i][1])
                    for i in range(WORLD)
                )
                all_ok &= ok_bits
                rec["results"][f"{name}_pdl{pdl}"] = {**st, "bit_ok": ok_bits}
                print(f"[pdlprobe] T={T} {name} pdl={pdl}: pair-median="
                      f"{st['median']:.3f}us p10={st['p10']:.3f} p90={st['p90']:.3f} "
                      f"(n={args.trials}; incl. predecessor) bit-ok={ok_bits}")
            a = rec["results"][f"{name}_pdl0"]["median"]
            b = rec["results"][f"{name}_pdl1"]["median"]
            print(f"[pdlprobe] T={T} {name}: pdl1/pdl0 pair-median delta = "
                  f"{(a - b):+.3f}us ({(a - b) / a * 100:+.2f}%)")
        emit(rec, args.out)
    return all_ok


def mode_ncusolo(args, ws) -> bool:
    """Single-rank pre-fed launch: NCU-safe profiling + compute-floor timing."""
    name = args.impls.split(",")[0]
    impl = load_impl(name)
    r = args.ncu_rank
    for row in load_rows(args):
        T = row["scalars"]["num_tokens"]
        io = RowIO(T, args.seed)
        outs, resouts = io.make_outputs()
        torch.cuda.set_device(r)
        # warm build/op registration
        ws.reset()
        _feed_solo_slots(ws, r, io, T)
        impl.launch(io.xs[r], outs[r], io.residuals[r], resouts[r], io.gammas[r],
                    ws.ranks[r], EPS, bool(args.pdl))
        torch.cuda.synchronize(r)
        # event-timed solo launches (reset + refeed outside the timed region)
        samples = []
        for _ in range(args.solo_iters):
            ws.reset()
            _feed_solo_slots(ws, r, io, T)
            torch.cuda.set_device(r)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            impl.launch(io.xs[r], outs[r], io.residuals[r], resouts[r], io.gammas[r],
                        ws.ranks[r], EPS, bool(args.pdl))
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000.0)
        st = stats(samples)
        print(f"[ncusolo] T={T} {name} rank{r}: solo-launch median={st['median']:.3f}us "
              f"mean={st['mean']:.3f} min={st['min']:.3f} (n={len(samples)}; "
              f"pre-fed slots, spin completes instantly -> compute floor)")
        emit({"mode": "ncusolo", "row": row["id"], "T": T, "impl": name, "rank": r,
              "solo_stats_us": st, "provenance": provenance(args)}, args.out)
    return True


# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True,
                    choices=["correctness", "bench", "noise", "stability", "selftest",
                             "ncusolo", "pdlprobe"])
    ap.add_argument("--ncu-rank", type=int, default=0)
    ap.add_argument("--solo-iters", type=int, default=30)
    ap.add_argument("--impls", default="fi", help="comma list: fi,jit (first = baseline)")
    ap.add_argument("--tokens", default="all", help="1 | 6 | all")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--warmup-replays", type=int, default=3)
    ap.add_argument("--correctness-rounds", type=int, default=3)
    ap.add_argument("--stability-replays", type=int, default=1000)
    ap.add_argument("--pdl", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(TASK_ROOT, "bench", "results.jsonl"))
    args = ap.parse_args()

    if args.mode == "selftest":
        cc.self_test()
        return

    ws = SpNvlsWorkspace(world_size=WORLD)
    try:
        ok = {"correctness": mode_correctness, "bench": mode_bench,
              "noise": mode_noise, "stability": mode_stability,
              "ncusolo": mode_ncusolo, "pdlprobe": mode_pdlprobe}[args.mode](args, ws)
    finally:
        ws.destroy()
    print(f"[harness] mode={args.mode} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
