// Parity test: CLIP vision encoder (C++/ggml) vs transformers fp32 reference.
//
// Fixtures from convert/parity_clip_vision.py into
// ${CMAKE_SOURCE_DIR}/benchmarks/fixtures/clip_vision/ (input.bin + embeds.bin).
// Runs the ggml graph on the same input and compares image_embeds.
// Missing fixtures -> SKIP_RETURN_CODE 77.
//
//   model: models/gguf/zero123pp_cond_f32.gguf (f32 weights for parity)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/backend.hpp"
#include "models/clip_vision.hpp"

#ifndef CLIPV_FIXTURE_DIR
#define CLIPV_FIXTURE_DIR "benchmarks/fixtures/clip_vision"
#endif
#ifndef CLIPV_MODEL
#define CLIPV_MODEL "models/gguf/zero123pp_cond_f32.gguf"
#endif

namespace {

std::vector<float> read_bin(const std::string & path, size_t expect) {
    std::FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes < 0 || (size_t) bytes != expect * sizeof(float)) { std::fclose(f); return {}; }
    std::vector<float> out(expect);
    if (std::fread(out.data(), sizeof(float), expect, f) != expect) { std::fclose(f); return {}; }
    std::fclose(f);
    return out;
}

} // namespace

int main() {
    const std::string dir = CLIPV_FIXTURE_DIR;
    const size_t n_in = 1 * 3 * 224 * 224;
    const size_t n_out = 1024;

    auto in_ref = read_bin(dir + "/input.bin", n_in);
    auto em_ref = read_bin(dir + "/embeds.bin", n_out);
    if (in_ref.empty() || em_ref.empty()) {
        std::printf("SKIP: fixtures missing under %s — run convert/parity_clip_vision.py\n", dir.c_str());
        return 77;
    }

    std::string err;
    std::string dev_name;
    ggml_backend_t backend = instantmesh::init_best_backend(dev_name, "cpu");
    instantmesh::ClipVisionModel model;
    if (!backend || !instantmesh::clip_vision_load(CLIPV_MODEL, backend, model, &err)) {
        std::printf("SKIP: cannot load %s: %s\n", CLIPV_MODEL, err.c_str());
        if (backend) ggml_backend_free(backend);
        return 77;
    }

    int dim = 0;
    float * out = instantmesh::clip_vision_encode(model, in_ref.data(), 1, &dim);
    if (!out || dim != (int) n_out) { std::printf("FAIL: bad encode output\n"); return 1; }

    double max_abs = 0, sum_abs = 0;
    for (size_t k = 0; k < n_out; ++k) {
        const double d = std::fabs((double) out[k] - em_ref[k]);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
    }
    std::printf("%s clip_vision embeds: max_abs=%.3e mean_abs=%.3e\n",
                max_abs < 1e-3 ? "PASS" : "FAIL", max_abs, sum_abs / n_out);
    free(out);
    return max_abs < 1e-3 ? 0 : 1;
}
