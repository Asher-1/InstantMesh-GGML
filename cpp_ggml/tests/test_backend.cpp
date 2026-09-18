// Verifies backend auto-detection, explicit device selection, and CPU fallback.
#include <cassert>
#include <cstdio>
#include <string>

#include "core/backend.hpp"

int main(int argc, char ** argv) {
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

    // Explicit device prefix selection (argv[1], e.g. cuda / vulkan): must
    // return a backend whose reported name mentions the requested prefix.
    if (argc > 1) {
        std::string dev_name;
        ggml_backend_t dev = instantmesh::init_best_backend(dev_name, argv[1]);
        assert(dev != nullptr);
        std::printf("device '%s' -> %s\n", argv[1], dev_name.c_str());
        for (auto & c : dev_name) c = (char) std::tolower(c);
        std::string want = argv[1];
        for (auto & c : want) c = (char) std::tolower(c);
        assert(dev_name.find(want) != std::string::npos);
        ggml_backend_free(dev);
    }

    ggml_backend_free(b);
    std::printf("test_backend: OK\n");
    return 0;
}
