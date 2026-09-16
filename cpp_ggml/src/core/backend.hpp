// Backend selection for the InstantMesh ggml pipeline.
//
// Inference code is backend-agnostic: at load time we query the ggml backend
// registry and pick the first GPU device (CUDA / Vulkan / Metal / ...), falling
// back to CPU. The build-time `-DGGML_CUDA=ON -DGGML_VULKAN=ON` options decide
// which backends are compiled in; this module just discovers what is available.
#ifndef INSTANTMESH_CORE_BACKEND_HPP
#define INSTANTMESH_CORE_BACKEND_HPP

#include <string>

#include "ggml-backend.h"

namespace instantmesh {

// Picks the best available compute backend for the pipeline.
//   device: nullptr/"auto" -> first GPU if any, else CPU.
//           "cpu"           -> force CPU.
//   The INSTANTMESH_DEVICE env var overrides "auto".
// Returns a newly-created backend (caller frees with ggml_backend_free) and
// writes a human-readable name into `name_out`.
ggml_backend_t init_best_backend(std::string & name_out, const char * device = nullptr);

// Free backend memory (VRAM) on the first GPU device; returns bytes, 0 if none.
size_t gpu_free_vram(void);

} // namespace instantmesh

#endif // INSTANTMESH_CORE_BACKEND_HPP