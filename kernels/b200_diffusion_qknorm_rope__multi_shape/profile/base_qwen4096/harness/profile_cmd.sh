# Profiling harness for run=base_qwen4096 impl=baseline case=qwen__4096 (B200, sm_100).
# The candidate CUDA extension is JIT-built with -lineinfo (src/wrapper.py).
CUDA_VISIBLE_DEVICES=0 KDA_RUN_CORRECTNESS=1 PROFILE_CASE=qwen__4096 PROFILE_IMPL=baseline \
  ncu --set full --profile-from-start off --target-processes all \
      -f -o profile/base_qwen4096/reports/full python profile_entry.py
CUDA_VISIBLE_DEVICES=0 KDA_RUN_CORRECTNESS=1 PROFILE_CASE=qwen__4096 PROFILE_IMPL=baseline \
  ncu --set source --section SourceCounters --profile-from-start off --target-processes all \
      -f -o profile/base_qwen4096/reports/source python profile_entry.py
