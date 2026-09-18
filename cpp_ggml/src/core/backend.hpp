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

// Explicit backend initialization options (replaces the former
// GGML_VK_DISABLE_F16=1 env switch for the zero123pp/InstantMesh pipelines).
struct BackendInitOptions {
    // Vulkan device fp16 shaders (storage+compute). Default false:
    // parity-safe — f16-weight matmuls stage f32 activations through f16 and
    // the rounding error compounds over sampling steps (zero123++ 75-step E2E
    // vs CPU: grid PSNR 54.9dB with fp16 off, 28.8dB with fp16 on). Set true
    // to trade exactness for ~9% speed. Latched at Vulkan device creation;
    // init_best_backend applies it before ggml_backend_dev_init, which is the
    // only requirement. Ignored by CPU/CUDA builds.
    bool vulkan_fp16 = false;
};

// Picks the best available compute backend for the pipeline.
//   device: nullptr/"auto" -> first GPU if any, else CPU.
//           "cpu"           -> force CPU.
//           "cuda"|"vulkan"|backend-name prefix -> first matching GPU device
//           (e.g. "cuda" matches CUDA0; "vulkan" matches Vulkan0).
//   The INSTANTMESH_DEVICE env var overrides "auto" only.
// Returns a newly-created backend (caller frees with ggml_backend_free) and
// writes a human-readable name into `name_out`.
ggml_backend_t init_best_backend(std::string & name_out, const char * device = nullptr,
                                 const BackendInitOptions & options = BackendInitOptions{});

// Free backend memory (VRAM) on the first GPU device; returns bytes, 0 if none.
size_t gpu_free_vram(void);

} // namespace instantmesh

#endif // INSTANTMESH_CORE_BACKEND_HPP