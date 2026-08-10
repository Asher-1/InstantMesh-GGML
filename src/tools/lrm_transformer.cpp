// TriplaneTransformer example: loads an lrm_transformer GGUF and runs one
// forward pass on a synthetic condition, printing output shape and summary
// stats. It is the second stage of the InstantMesh geometry pipeline (input is
// the DINO encoder output).
//
//   ./build/lrm_transformer --device auto models/gguf/lrm_transformer_f16.gguf
//
// For parity: pass `--cond cond.bin --cond-len N --out triplane.bin`.
// cond.bin holds [B, L_cond, 768] f32 image features; triplane.bin gets the
// [B, 3, 80, 64, 64] f32 output. See convert/parity_lrm.py for the reference.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "core/backend.hpp"
#include "models/lrm_transformer.hpp"

static void usage(const char * p) {
    std::fprintf(stderr,
        "usage: %s [--device auto|cpu|Vulkan] <lrm_transformer.gguf>\n"
        "       [--batch B] [--cond-len L] [--cond cond.bin] [--out triplane.bin]\n", p);
}

static bool read_blob(const char * path, std::vector<float> & out) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "failed to open %s\n", path); return false; }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<float> tmp((size_t) n / sizeof(float));
    if (std::fread(tmp.data(), sizeof(float), tmp.size(), f) != tmp.size()) { std::fclose(f); return false; }
    std::fclose(f); out.swap(tmp); return true;
}

int main(int argc, char ** argv) {
    const char * device = "auto";
    const char * model = nullptr;
    const char * cond_blob = nullptr, * out_blob = nullptr;
    int B = 1, cond_len = 197;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = argv[++i];
        else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) B = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--cond-len") == 0 && i + 1 < argc) cond_len = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--cond") == 0 && i + 1 < argc) cond_blob = argv[++i];
        else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_blob = argv[++i];
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) { usage(argv[0]); return 0; }
        else if (argv[i][0] != '-') model = argv[i];
        else { usage(argv[0]); return 1; }
    }
    if (!model) { usage(argv[0]); return 1; }

    std::string backend_name;
    ggml_backend_t backend = instantmesh::init_best_backend(backend_name, device);
    std::printf("backend: %s\n", backend_name.c_str());

    instantmesh::LrmTransformerModel tr;
    std::string error;
    if (!instantmesh::lrm_transformer_load(model, backend, tr, &error)) {
        std::fprintf(stderr, "failed to load lrm_transformer: %s\n", error.c_str());
        ggml_backend_free(backend);
        return 1;
    }
    const auto & hp = tr.hp;
    std::printf("lrm_transformer: inner=%d layers=%d heads=%d cond=%d low=%d high=%d dim=%d\n",
        hp.inner_dim, hp.num_layers, hp.num_heads, hp.cond_dim,
        hp.triplane_low_res, hp.triplane_high_res, hp.triplane_dim);

    // inputs: condition from blob or synthetic.
    std::vector<float> cond((size_t) B * cond_len * hp.cond_dim);
    if (cond_blob) {
        std::vector<float> blob;
        if (!read_blob(cond_blob, blob)) return 1;
        if ((long) blob.size() != (long) cond.size()) {
            std::fprintf(stderr, "cond.bin size mismatch: got %zu, need %zu\n",
                blob.size(), cond.size());
            return 1;
        }
        std::memcpy(cond.data(), blob.data(), cond.size() * sizeof(float));
    } else {
        for (size_t i = 0; i < cond.size(); ++i) cond[i] = ((i % 10) - 5) * 0.05f;
    }

    int planes = 0, dim = 0, out_h = 0, out_w = 0;
    float * out = instantmesh::lrm_transformer_forward(tr, cond.data(), B, cond_len,
                                                       &planes, &dim, &out_h, &out_w);
    std::printf("output: [%d, %d, %d, %d, %d]\n", B, planes, dim, out_h, out_w);
    if (out_blob) {
        FILE * f = std::fopen(out_blob, "wb");
        size_t n = (size_t) B * planes * dim * out_h * out_w;
        if (f) { std::fwrite(out, sizeof(float), n, f); std::fclose(f); }
        std::printf("wrote %s (%zu bytes)\n", out_blob, n * sizeof(float));
    }
    float mean = 0, var = 0, n = (float) B * planes * dim * out_h * out_w;
    for (long i = 0; i < (long) (B * planes * dim * out_h * out_w); ++i) { mean += out[i]; var += out[i] * out[i]; }
    mean /= n; var = var / n - mean * mean;
    std::printf("stats: mean=%.5f std=%.5f  first=%.5f\n", mean, std::sqrt(var > 0 ? var : 0), out[0]);
    std::free(out);

    ggml_backend_free(backend);
    return 0;
}