// Parity test: AutoencoderKL encode/decode (C++/ggml) vs diffusers fp32.
//
// Fixtures from convert/parity_vae.py (run under /tmp/vref python) into
// ${CMAKE_SOURCE_DIR}/benchmarks/fixtures/vae/. Missing fixtures -> SKIP 77.
//
// Thresholds: the encoder uses F32-im2col convs (conv_f32) so its residual
// vs torch is pure f32 summation-order noise (~2e-4 on the latents after a
// 23-conv stack; a systematic bug shows up at >=1e-2). The decoder keeps
// ggml_conv_2d's internal F16 im2col: errors accumulate over the 3x-upsample
// stack to a stable ~3e-3 absolute on a [-0.7,0.7] output — invisible at 8-bit
// (1 RGB code ≈ 2.45e-3) and ~50dB PSNR vs the f32 reference.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "core/backend.hpp"
#include "models/vae.hpp"

#ifndef VAE_FIXTURE_DIR
#define VAE_FIXTURE_DIR "benchmarks/fixtures/vae"
#endif
#ifndef VAE_MODEL
#define VAE_MODEL "models/gguf/zero123pp_vae_f32.gguf"
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
    const std::string dir = VAE_FIXTURE_DIR;
    auto enc_in  = read_bin(dir + "/enc_in.bin", (size_t) 1 * 3 * 512 * 512);
    auto enc_lat = read_bin(dir + "/enc_lat.bin", (size_t) 1 * 4 * 64 * 64);
    auto dec_lat = read_bin(dir + "/dec_lat.bin", (size_t) 1 * 4 * 80 * 120);
    auto dec_img = read_bin(dir + "/dec_img.bin", (size_t) 1 * 3 * 640 * 960);
    if (enc_in.empty() || enc_lat.empty() || dec_lat.empty() || dec_img.empty()) {
        std::printf("SKIP: fixtures missing under %s — run convert/parity_vae.py (venv)\n", dir.c_str());
        return 77;
    }

    std::string err, dev;
    ggml_backend_t backend = instantmesh::init_best_backend(dev, "cpu");
    instantmesh::VaeModel model;
    if (!backend || !instantmesh::vae_load(VAE_MODEL, backend, model, &err)) {
        std::printf("SKIP: cannot load %s: %s\n", VAE_MODEL, err.c_str());
        if (backend) ggml_backend_free(backend);
        return 77;
    }

    int fails = 0;
    {
        int lh = 0, lw = 0;
        float * lat = instantmesh::vae_encode_mode(model, enc_in.data(), 1, 512, 512, &lh, &lw);
        double m = 0;
        for (size_t k = 0; k < (size_t) 1 * 4 * 64 * 64; ++k)
            m = std::max(m, (double) std::fabs(lat[k] - enc_lat[k]));
        std::printf("%s vae encode mode: max_abs=%.3e\n", m < 5e-4 ? "PASS" : "FAIL", m);
        if (!(m < 5e-4)) ++fails;
        free(lat);
    }
    {
        int oh = 0, ow = 0;
        float * img = instantmesh::vae_decode(model, dec_lat.data(), 1, 80, 120, &oh, &ow);
        double m = 0, sum = 0;
        const size_t n = (size_t) 1 * 3 * 640 * 960;
        for (size_t k = 0; k < n; ++k) {
            const double d = std::fabs((double) img[k] - dec_img[k]);
            m = std::max(m, d);
            sum += d;
        }
        std::printf("%s vae decode: max_abs=%.3e mean_abs=%.3e\n",
                    m < 5e-3 ? "PASS" : "FAIL", m, sum / n);
        if (!(m < 5e-3)) ++fails;
        free(img);
    }
    ggml_backend_free(backend);
    std::printf("%s\n", fails == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return fails == 0 ? 0 : 1;
}
