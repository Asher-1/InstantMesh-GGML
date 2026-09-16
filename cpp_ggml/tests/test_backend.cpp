// Verifies backend auto-detection and CPU fallback.
#include <cassert>
#include <cstdio>
#include <string>

#include "core/backend.hpp"

int main() {
    // auto: must return a working backend (GPU or CPU) with a non-empty name.
    std::string name;
    ggml_backend_t b = instantmesh::init_best_backend(name, nullptr);
    assert(b != nullptr);
    assert(!name.empty());
    std::printf("auto backend: %s\n", name.c_str());

    // cpu: must always succeed and report "CPU".
    std::string cpu_name;
    ggml_backend_t cpu = instantmesh::init_best_backend(cpu_name, "cpu");
    assert(cpu != nullptr);
    assert(cpu_name == "CPU");
    ggml_backend_free(cpu);

    ggml_backend_free(b);
    std::printf("test_backend: OK\n");
    return 0;
}