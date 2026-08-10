#include "core/backend.hpp"

#include <cstdlib>
#include <thread>

#include "ggml-cpu.h"

namespace instantmesh {

ggml_backend_t init_best_backend(std::string & name_out, const char * device) {
    std::string want = device ? device : "";
    if (want.empty() || want == "auto") {
        if (const char * env = std::getenv("INSTANTMESH_DEVICE")) {
            want = env;
        }
    }

    // Prefer a GPU device exposed by the registry (CUDA / Vulkan / Metal / ...).
    if (want != "cpu") {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
                ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
                if (b) {
                    const char * d = ggml_backend_dev_description(dev);
                    name_out = d ? d : ggml_backend_dev_name(dev);
                    return b;
                }
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