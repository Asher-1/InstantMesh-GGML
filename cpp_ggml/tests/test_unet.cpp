// Parity test: UNet RefOnly double-forward (C++/ggml) vs the official
// zero123plus ReferenceOnlyAttnProc reference (fp32, fixtures from
// convert/dump_unet_stages.py). Weights are the f16 GGUF, so the residual
// vs an fp32 reference is dominated by f16 weight quantization (~1e-2
// relative through the deep stack); structural bugs blow up way past that.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/backend.hpp"
#include "models/unet.hpp"

#ifndef UNET_FIXTURE_DIR
#define UNET_FIXTURE_DIR "benchmarks/fixtures/unet"
#endif
#ifndef UNET_MODEL
#define UNET_MODEL "models/gguf/zero123pp_unet_f16.gguf"
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
    const std::string dir = UNET_FIXTURE_DIR;
    const int B = 2, H = 80, W = 120, L = 77;
    const size_t lat_n = (size_t) B * 4 * H * W;
    const size_t ctx_n = (size_t) B * L * 1024;

    auto in    = read_bin(dir + "/unet_in.bin", lat_n);
    auto refin = read_bin(dir + "/unet_ref_in.bin", lat_n);
    auto ctx   = read_bin(dir + "/unet_ctx.bin", ctx_n);
    auto tref  = read_bin(dir + "/unet_t.bin", 1);
    auto out   = read_bin(dir + "/unet_eps_out.bin", lat_n);
    if (in.empty() || refin.empty() || ctx.empty() || tref.empty() || out.empty()) {
        std::printf("SKIP: fixtures missing under %s — run convert/dump_unet_stages.py (venv)\n", dir.c_str());
        return 77;
    }

    std::string err, dev;
    ggml_backend_t backend = instantmesh::init_best_backend(dev, "cpu");
    instantmesh::UnetModel model;
    if (!backend || !instantmesh::unet_load(UNET_MODEL, backend, model, &err)) {
        std::printf("SKIP: cannot load %s: %s\n", UNET_MODEL, err.c_str());
        if (backend) ggml_backend_free(backend);
        return 77;
    }

    int oh = 0, ow = 0;
    float * eps = instantmesh::unet_forward_refonly(model, in.data(), refin.data(),
                                                    ctx.data(), L, B, H, W,
                                                    tref[0], &oh, &ow);
    double m = 0, sum = 0;
    for (size_t k = 0; k < lat_n; ++k) {
        const double d = std::fabs((double) eps[k] - out[k]);
        m = std::max(m, d);
        sum += d;
    }
    const double mean = sum / lat_n;
    std::printf("%s unet refonly double-forward: max_abs=%.3e mean_abs=%.3e (n=%zu)\n",
                m < 0.35 ? "PASS" : "FAIL", m, mean, lat_n);
    free(eps);
    ggml_backend_free(backend);
    return m < 0.35 ? 0 : 1;
}
