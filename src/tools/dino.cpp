// DINO encoder example: loads a dino GGUF and runs one forward pass on a
// synthetic image + camera, printing output shape and a few summary stats.
//
//   ./build/dino --device auto models/gguf/dino_f16.gguf
//
// For parity: pass `--in inputs.bin --out features.bin`. inputs.bin holds
// [B,3,H,W] f32 image followed by [B,16] f32 camera (exactly what dino_encode
// consumes); features.bin gets the [B,1+N,768] f32 output. See
// convert/parity_dino.py for the PyTorch reference.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "core/backend.hpp"
#include "models/dino.hpp"

static void usage(const char * p) {
    std::fprintf(stderr,
        "usage: %s [--device auto|cpu|Vulkan] <dino.gguf>\n"
        "       [--size H W] [--batch B] [--in inputs.bin] [--out features.bin]\n", p);
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
    const char * in_blob = nullptr, * out_blob = nullptr;
    int H = 224, W = 224, B = 1;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = argv[++i];
        else if (std::strcmp(argv[i], "--size") == 0 && i + 2 < argc) { H = std::atoi(argv[++i]); W = std::atoi(argv[++i]); }
        else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) B = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--in") == 0 && i + 1 < argc) in_blob = argv[++i];
        else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_blob = argv[++i];
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) { usage(argv[0]); return 0; }
        else if (argv[i][0] != '-') model = argv[i];
        else { usage(argv[0]); return 1; }
    }
    if (!model) { usage(argv[0]); return 1; }

    std::string backend_name;
    ggml_backend_t backend = instantmesh::init_best_backend(backend_name, device);
    std::printf("backend: %s\n", backend_name.c_str());

    instantmesh::DinoModel dino;
    std::string error;
    if (!instantmesh::dino_load(model, backend, dino, &error)) {
        std::fprintf(stderr, "failed to load dino: %s\n", error.c_str());
        ggml_backend_free(backend);
        return 1;
    }
    std::printf("dino: hidden=%d layers=%d heads=%d patch=%d\n",
        dino.hp.hidden_size, dino.hp.num_hidden_layers,
        dino.hp.num_attention_heads, dino.hp.patch_size);

    // inputs: from blob (image then camera) or synthetic.
    std::vector<float> image((size_t) B * 3 * H * W), camera((size_t) B * 16);
    if (in_blob) {
        std::vector<float> blob;
        if (!read_blob(in_blob, blob)) return 1;
        if (blob.size() != image.size() + camera.size()) {
            std::fprintf(stderr, "inputs.bin size mismatch: got %zu, need %zu\n",
                blob.size(), image.size() + camera.size());
            return 1;
        }
        std::memcpy(image.data(), blob.data(), image.size() * sizeof(float));
        std::memcpy(camera.data(), blob.data() + image.size(), camera.size() * sizeof(float));
    } else {
        for (auto & v : image) v = (std::rand() % 1000) / 1000.0f;
        for (int b = 0; b < B; ++b) for (int i = 0; i < 16; ++i) camera[b * 16 + i] = (i % 4) * 0.1f;
    }

    int seq = 0, hidden = 0;
    float * out = instantmesh::dino_encode(dino, image.data(), camera.data(),
                                           B, H, W, &seq, &hidden);
    std::printf("output: [%d, %d, %d]\n", B, seq, hidden);
    if (out_blob) {
        FILE * f = std::fopen(out_blob, "wb");
        if (f) { std::fwrite(out, sizeof(float), (size_t) B * seq * hidden, f); std::fclose(f); }
        std::printf("wrote %s (%zu bytes)\n", out_blob, (size_t) B * seq * hidden * sizeof(float));
    }
    float mean = 0, var = 0, n = (float)(B * seq * hidden);
    for (long i = 0; i < (long)(B * seq * hidden); ++i) { mean += out[i]; var += out[i] * out[i]; }
    mean /= n; var = var / n - mean * mean;
    std::printf("stats: mean=%.5f std=%.5f  first_cls=%.5f\n", mean, std::sqrt(var > 0 ? var : 0), out[0]);
    std::free(out);

    ggml_backend_free(backend);
    return 0;
}