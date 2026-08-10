// Geometry-prediction example: loads a synthesizer GGUF and runs the triplane
// sampling + OSGDecoder on a voxel grid, printing sdf/deformation/weight
// summary stats. Third stage of the InstantMesh geometry pipeline (input is
// the TriplaneTransformer output).
//
//   ./build/synthesizer --device auto models/gguf/synthesizer_f32.gguf \
//       --planes planes.bin --verts verts.bin --cubes cubes.bin \
//       --in-dir /tmp/synth_ref --out-dir /tmp/synth_cpp
//
// For parity: pass --in-dir (verts.bin/cubes.bin) and --out-dir to dump
// sdf/deformation/weight binaries. See convert/parity_synth.py for the
// reference.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "core/backend.hpp"
#include "models/synthesizer.hpp"

static void usage(const char * p) {
    std::fprintf(stderr,
        "usage: %s [--device auto|cpu|Vulkan] <synthesizer.gguf>\n"
        "       [--batch N] [--plane-h H] [--plane-w W]\n"
        "       [--planes planes.bin] [--in-dir dir] [--out-dir dir]\n", p);
}

static bool read_blob(const char * path, std::vector<float> & out) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "failed to open %s\n", path); return false; }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<float> tmp((size_t) n / sizeof(float));
    if (std::fread(tmp.data(), sizeof(float), tmp.size(), f) != tmp.size()) { std::fclose(f); return false; }
    std::fclose(f); out.swap(tmp); return true;
}

static bool read_ints(const char * path, std::vector<int32_t> & out) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "failed to open %s\n", path); return false; }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<int32_t> tmp((size_t) n / sizeof(int32_t));
    if (std::fread(tmp.data(), sizeof(int32_t), tmp.size(), f) != tmp.size()) { std::fclose(f); return false; }
    std::fclose(f); out.swap(tmp); return true;
}

int main(int argc, char ** argv) {
    const char * device = "auto";
    const char * model = nullptr;
    const char * planes_path = nullptr, * in_dir = nullptr, * out_dir = nullptr;
    int N = 1, H = 8, W = 8;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = argv[++i];
        else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) N = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--plane-h") == 0 && i + 1 < argc) H = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--plane-w") == 0 && i + 1 < argc) W = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--planes") == 0 && i + 1 < argc) planes_path = argv[++i];
        else if (std::strcmp(argv[i], "--in-dir") == 0 && i + 1 < argc) in_dir = argv[++i];
        else if (std::strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) out_dir = argv[++i];
        else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) { usage(argv[0]); return 0; }
        else if (argv[i][0] != '-') model = argv[i];
        else { usage(argv[0]); return 1; }
    }
    if (!model) { usage(argv[0]); return 1; }

    std::string backend_name;
    ggml_backend_t backend = instantmesh::init_best_backend(backend_name, device);
    std::printf("backend: %s\n", backend_name.c_str());

    instantmesh::SynthesizerModel syn;
    std::string error;
    if (!instantmesh::synthesizer_load(model, backend, syn, &error)) {
        std::fprintf(stderr, "failed to load synthesizer: %s\n", error.c_str());
        ggml_backend_free(backend);
        return 1;
    }
    const auto & hp = syn.hp;
    std::printf("synthesizer: plane_dim=%d hidden=%d layers=%d\n",
        hp.plane_dim, hp.hidden, hp.num_layers);

    // grid geometry (verts + cubes) from the reference dir.
    std::vector<float> verts;
    std::vector<int32_t> cubes;
    if (in_dir) {
        std::string v = std::string(in_dir) + "/verts.bin";
        std::string c = std::string(in_dir) + "/cubes.bin";
        if (!read_blob(v.c_str(), verts)) return 1;
        if (!read_ints(c.c_str(), cubes)) return 1;
    } else {
        std::fprintf(stderr, "--in-dir required (holds verts.bin/cubes.bin)\n");
        return 1;
    }
    int M = (int) verts.size() / 3;
    int n_cubes = (int) cubes.size() / 8;
    std::printf("grid: M=%d n_cubes=%d\n", M, n_cubes);

    // planes input.
    std::vector<float> planes((size_t) N * 3 * hp.plane_dim * H * W);
    if (planes_path) {
        std::vector<float> blob;
        if (!read_blob(planes_path, blob)) return 1;
        if ((long) blob.size() != (long) planes.size()) {
            std::fprintf(stderr, "planes.bin size mismatch: got %zu, need %zu\n",
                blob.size(), planes.size());
            return 1;
        }
        std::memcpy(planes.data(), blob.data(), planes.size() * sizeof(float));
    } else {
        for (size_t i = 0; i < planes.size(); ++i) planes[i] = ((i % 10) - 5) * 0.05f;
    }

    float * sdf = nullptr, * deformation = nullptr, * weight = nullptr, * rgb = nullptr;
    instantmesh::synthesizer_forward(syn, planes.data(), N, H, W,
                                     verts.data(), M, cubes.data(), n_cubes,
                                     &sdf, &deformation, &weight);
    if (!instantmesh::synthesizer_texture_forward(syn, planes.data(), N, H, W,
                                                  verts.data(), M, &rgb)) {
        std::fprintf(stderr, "rgb branch failed\n");
        return 1;
    }

    if (out_dir) {
        size_t ns = (size_t) N * M, nd = (size_t) N * M * 3, nw = (size_t) N * n_cubes * 21;
        FILE * f;
        f = std::fopen((std::string(out_dir) + "/sdf.bin").c_str(), "wb");
        if (f) { std::fwrite(sdf, sizeof(float), ns, f); std::fclose(f); }
        f = std::fopen((std::string(out_dir) + "/deformation.bin").c_str(), "wb");
        if (f) { std::fwrite(deformation, sizeof(float), nd, f); std::fclose(f); }
        f = std::fopen((std::string(out_dir) + "/weight.bin").c_str(), "wb");
        if (f) { std::fwrite(weight, sizeof(float), nw, f); std::fclose(f); }
        // rgb.bin = raw sigmoid(net_rgb) — no MipNeRF affine clamp (matches
        // the pytorch parity reference torch.sigmoid(net_rgb)).
        f = std::fopen((std::string(out_dir) + "/rgb.bin").c_str(), "wb");
        if (f) { std::fwrite(rgb, sizeof(float), (size_t) N * M * 3, f); std::fclose(f); }
        std::printf("wrote %s (sdf/deformation/weight/rgb)\n", out_dir);
    }

    auto stats = [](const float * p, long n) {
        double mean = 0, var = 0;
        for (long i = 0; i < n; ++i) { mean += p[i]; var += (double) p[i] * p[i]; }
        mean /= n; var = var / n - mean * mean;
        std::printf("mean=%.5f std=%.5f first=%.5f\n", mean, std::sqrt(var > 0 ? var : 0), p[0]);
    };
    std::printf("sdf [");        stats(sdf, (long) N * M);
    std::printf("deform [");     stats(deformation, (long) N * M * 3);
    std::printf("weight [");     stats(weight, (long) N * n_cubes * 21);

    std::free(sdf); std::free(deformation); std::free(weight); std::free(rgb);
    ggml_backend_free(backend);
    return 0;
}