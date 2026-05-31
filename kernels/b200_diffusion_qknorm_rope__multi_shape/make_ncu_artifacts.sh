#!/bin/bash
# Build the ncu-report-skill artifact layout for the final v3 candidate and the
# baseline: profile/<run>/{harness,reports,analysis}/ with both `--set full` and
# `--set source --section SourceCounters` reports, plus parsed analysis CSVs.
# Run inside sglang_bbuf on ion-b200 from the task workspace, idle GPU pinned.
set -u
GPU="${CUDA_VISIBLE_DEVICES:-0}"
echo "GPU0 before: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$GPU")"

run_one() {
  local run="$1" impl="$2" case="$3"
  echo "=== $run (impl=$impl case=$case) ==="
  mkdir -p "profile/$run/harness" "profile/$run/reports" "profile/$run/analysis"
  cp profile_entry.py "profile/$run/harness/profile_entry.py"
  cat > "profile/$run/harness/profile_cmd.sh" <<CMD
# Profiling harness for run=$run impl=$impl case=$case (B200, sm_100).
# The candidate CUDA extension is JIT-built with -lineinfo (src/wrapper.py).
CUDA_VISIBLE_DEVICES=$GPU KDA_RUN_CORRECTNESS=1 PROFILE_CASE=$case PROFILE_IMPL=$impl \\
  ncu --set full --profile-from-start off --target-processes all \\
      -f -o profile/$run/reports/full python profile_entry.py
CUDA_VISIBLE_DEVICES=$GPU KDA_RUN_CORRECTNESS=1 PROFILE_CASE=$case PROFILE_IMPL=$impl \\
  ncu --set source --section SourceCounters --profile-from-start off --target-processes all \\
      -f -o profile/$run/reports/source python profile_entry.py
CMD
  CUDA_VISIBLE_DEVICES="$GPU" KDA_RUN_CORRECTNESS=1 TORCH_CUDA_ARCH_LIST=10.0 PROFILE_CASE="$case" PROFILE_IMPL="$impl" \
    ncu --set full --profile-from-start off --target-processes all -f -o "profile/$run/reports/full" python profile_entry.py >/dev/null 2>&1
  CUDA_VISIBLE_DEVICES="$GPU" KDA_RUN_CORRECTNESS=1 TORCH_CUDA_ARCH_LIST=10.0 PROFILE_CASE="$case" PROFILE_IMPL="$impl" \
    ncu --set source --section SourceCounters --profile-from-start off --target-processes all -f -o "profile/$run/reports/source" python profile_entry.py >/dev/null 2>&1
  # Parsed analysis: SOL/occupancy/limiter summary + full metric dump + source-counter stalls.
  ncu --import "profile/$run/reports/full.ncu-rep" --page details 2>/dev/null \
    | grep -iE "Duration|Compute \(SM\)|Memory Throughput|DRAM Throughput|Achieved Occupancy|Registers Per Thread|Grid Size|Waves Per|Block Limit|excessive sectors|Est. Speedup|stall" \
    > "profile/$run/analysis/sol_summary.txt"
  ncu --import "profile/$run/reports/full.ncu-rep" --csv --page raw 2>/dev/null > "profile/$run/analysis/full_metrics.csv" || true
  ncu --import "profile/$run/reports/source.ncu-rep" --page source 2>/dev/null | head -120 > "profile/$run/analysis/source_counters.txt" || true
  # Per-run REPORT.md for every profiled run (preserve a hand-authored one if present).
  if [ ! -f "profile/$run/REPORT.md" ]; then
    {
      echo "# NCU REPORT — $run (impl=$impl, case=$case), B200 sm_100"
      echo
      echo "Auto-generated. Raw: reports/{full,source}.ncu-rep; harness/; analysis/."
      echo "Name the active limiter from the Speed-of-Light below + analysis/source_counters.txt."
      echo
      echo '## Speed-of-Light (analysis/sol_summary.txt)'
      echo '```'
      cat "profile/$run/analysis/sol_summary.txt"
      echo '```'
    } > "profile/$run/REPORT.md"
    echo "  wrote auto REPORT.md"
  else
    echo "  kept existing REPORT.md"
  fi
  echo "  done: $(ls -1 profile/$run/reports/ | tr '\n' ' ')"
}

run_one cand_qwen4096_v3 candidate qwen__4096   # large bucket, candidate v3
run_one base_qwen4096    baseline  qwen__4096   # large bucket, SGLang baseline
run_one cand_qwen19_v3   candidate qwen__19     # tiny bucket, candidate v3
echo "GPU0 after: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "$GPU")"
echo NCU_ARTIFACTS_DONE
