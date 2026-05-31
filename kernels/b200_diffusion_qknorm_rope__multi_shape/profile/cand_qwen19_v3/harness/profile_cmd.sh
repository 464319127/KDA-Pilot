# Profiling harness for run=cand_qwen19_v3 impl=candidate case=qwen__19 (B200, sm_100).
# The candidate CUDA extension is JIT-built with -lineinfo (src/wrapper.py).
CUDA_VISIBLE_DEVICES=0 KDA_RUN_CORRECTNESS=1 PROFILE_CASE=qwen__19 PROFILE_IMPL=candidate \
  ncu --set full --profile-from-start off --target-processes all \
      -f -o profile/cand_qwen19_v3/reports/full python profile_entry.py
CUDA_VISIBLE_DEVICES=0 KDA_RUN_CORRECTNESS=1 PROFILE_CASE=qwen__19 PROFILE_IMPL=candidate \
  ncu --set source --section SourceCounters --profile-from-start off --target-processes all \
      -f -o profile/cand_qwen19_v3/reports/source python profile_entry.py
