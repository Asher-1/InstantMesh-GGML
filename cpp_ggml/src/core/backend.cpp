#include "core/backend.hpp"

#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <thread>

#include "ggml-cpu.h"

namespace instantmesh {

static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

// Explicit fp16 switch provided by the patched ggml-vulkan backend.
// Weak-declared so CPU-only/CUDA-only builds still link; a null symbol is a no-op.
#if defined(__GNUC__) && !defined(_WIN32)
extern "C" void ggml_backend_vk_set_fp16(bool fp16) __attribute__((weak));
static void set_vulkan_fp16(bool fp16) {
    if (ggml_backend_vk_set_fp16) {
        ggml_backend_vk_set_fp16(fp16);
    }
}
#else
static void set_vulkan_fp16(bool) {}
#endif

// Backend selection for the InstantMesh ggml pipeline.
//
//   device: nullptr/"auto" -> first GPU if any, else CPU.
//           "cpu"           -> force CPU.
//           "cuda"|"vulkan"|backend-name prefix -> first matching GPU device.
//   The INSTANTMESH_DEVICE env var overrides "auto" only.
ggml_backend_t init_best_backend(std::string & name_out, const char * device,
                                 const BackendInitOptions & options) {
    // The Vulkan device latches its fp16 capability once at creation
    // (ggml_vk_get_device, reached via ggml_backend_dev_init below); registry
    // enumeration (dev_count/dev_get) does not create the device. Applying
    // the option here — before any dev_init — is the only timing requirement.
    set_vulkan_fp16(options.vulkan_fp16);
    std::string want = device ? device : "";
    if (want.empty() || want == "auto") {
        if (const char * env = std::getenv("INSTANTMESH_DEVICE")) {
            want = env;
        }
    }

    const std::string want_l = to_lower(want);
    const bool pick_all = want.empty() || want_l == "auto";

    // Prefer a GPU device exposed by the registry (CUDA / Vulkan / Metal / ...).
    // An explicit device string (e.g. "cuda", "vulkan") restricts the match to
    // devices whose name/description starts with it; otherwise the first GPU wins.
    if (!pick_all || want_l != "cpu") {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
            const char * dev_name = ggml_backend_dev_name(dev);
            if (!pick_all) {
                const std::string dev_l = to_lower(dev_name ? dev_name : "");
                if (dev_l.rfind(want_l, 0) != 0) continue;
            }
            ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
            if (b) {
                const char * d = ggml_backend_dev_description(dev);
                name_out = d ? d : dev_name;
                return b;
            }
        }
    }

    name_out = "CPU";
    ggml_backend_t cpu = ggml_backend_cpu_init();
    int n_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (const char * env = std::getenv("INSTANTMESH_N_THREADS")) {
        const int v = std::atoi(env);
        if (v > 0) n_threads = v;
    }
    if (n_threads > 0) ggml_backend_cpu_set_n_threads(cpu, n_threads);
    return cpu;
}

size_t gpu_free_vram(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) return 0; // CPU-only build/host
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    return free;
}

} // namespace instantmesh
