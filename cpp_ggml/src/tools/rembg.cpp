// rembg — AI background removal (BiRefNet / RMBG-2.0, GGUF weights).
//
// Official InstantMesh uses the `rembg` pip package (u2net) in run.py Stage 1;
// this tool aligns that capability in pure C++/ggml with the stronger
// RMBG-2.0 (BiRefNet) model, vendored at third_party/RMBG-2.0-GGML.
//
//   usage: rembg --model models/gguf/rmbg_f16.gguf --input in.png --out out.png
//                [--device auto|cpu|gpu]
//
// Input: PNG/JPEG bytes (any resolution, RGB or RGBA).
// Output: RGBA PNG with the background removed (alpha = soft matte).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rmbg_capi.h"

namespace {

void usage(const char * p) {
    std::fprintf(stderr,
        "usage: %s --model <rmbg.gguf> --input <image> --out <rgba.png>\n"
        "       [--device auto|cpu|gpu]\n",
        p);
}

std::vector<uint8_t> read_file(const char * path) {
    std::FILE * f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf((size_t) n);
    if (n > 0 && std::fread(buf.data(), 1, (size_t) n, f) != (size_t) n) {
        std::fprintf(stderr, "short read on %s\n", path); std::exit(1);
    }
    std::fclose(f);
    return buf;
}

} // namespace

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * input_path = nullptr;
    const char * out_path = nullptr;
    const char * device = "auto";

    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--input") && i + 1 < argc) input_path = argv[++i];
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) device = argv[++i];
        else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (!model_path || !input_path || !out_path) { usage(argv[0]); return 1; }

    char err[512] = {0};
    rmbg_model * m = rmbg_load(model_path, device, err, (int) sizeof(err));
    if (!m) { std::fprintf(stderr, "rmbg_load failed: %s\n", err); return 1; }

    std::vector<uint8_t> img = read_file(input_path);
    uint8_t * out_png = nullptr;
    int out_len = 0;
    int rc = rmbg_remove_background(m, img.data(), (int) img.size(),
                                    &out_png, &out_len, err, (int) sizeof(err));
    if (rc != 0) { std::fprintf(stderr, "rmbg_remove_background failed: %s\n", err); rmbg_free(m); return 1; }

    std::FILE * f = std::fopen(out_path, "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", out_path); rmbg_free_buffer(out_png); rmbg_free(m); return 1; }
    std::fwrite(out_png, 1, (size_t) out_len, f);
    std::fclose(f);
    std::printf("rembg: %s -> %s (%d bytes rgba png)\n", input_path, out_path, out_len);

    rmbg_free_buffer(out_png);
    rmbg_free(m);
    return 0;
}
