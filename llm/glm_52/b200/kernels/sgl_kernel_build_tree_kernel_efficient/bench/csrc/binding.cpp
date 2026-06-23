// Single pybind module exposing the recovered baseline, the native-CUDA
// candidate, and an empty-kernel launch-floor probe through ONE identical build
// (same sources compiled together, same flags) so baseline and candidate share
// the exact same registration/export/build style and call overhead.
#include <torch/extension.h>

#include "build_tree_ext.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_tree_baseline", &build_tree_baseline, "Recovered upstream baseline (in place, returns None)");
  m.def("build_tree_candidate", &build_tree_candidate, "Native-CUDA candidate (in place, returns None)");
  m.def("build_tree_noop", &build_tree_noop, "Empty-kernel launch-floor probe");
}
