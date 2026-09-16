// Standalone bring-up probe: real GGUF conv_in weight + fixture input through
// a MINIMAL conv_2d graph, compared element-wise against an in-process naive
// f32 conv2d. Isolates the op from the full VAE graph (gallocr/plan pressure).
#include <cmath>
#include <cstdio>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

int main(int argc, char ** argv) {
    const char * gguf_path = argc > 1 ? argv[1] : "models/gguf/zero123pp_vae_f32.gguf";
    const char * fixture   = argc > 2 ? argv[2] : "benchmarks/fixtures/vae/enc_in.bin";
    const int W = 512, H = 512, IC = 3, OC = 128;
    ggml_context * wctx_holder = nullptr;

    // fixture (torch [3,H,W] bytes)
    std::vector<float> image((size_t) IC * H * W);
    {
        FILE * f = std::fopen(fixture, "rb");
        if (!f || std::fread(image.data(), 4, image.size(), f) != image.size()) {
            std::printf("fixture read failed\n"); return 1;
        }
        std::fclose(f);
    }

    // load only the conv_in weight + bias from the GGUF
    gguf_init_params ip; ip.no_alloc = true; ip.ctx = &wctx_holder;
    gguf_context * gu = gguf_init_from_file(gguf_path, ip);
    if (!gu) { std::printf("gguf init failed\n"); return 1; }
    ggml_context * wctx = wctx_holder;
    struct ggml_tensor * k_in = ggml_get_tensor(wctx, "encoder.conv_in.weight");
    struct ggml_tensor * b_in = ggml_get_tensor(wctx, "encoder.conv_in.bias");
    if (!k_in) { std::printf("tensor not found\n"); return 1; }
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_alloc_ctx_tensors(wctx, be);
    {
        FILE * f = std::fopen(gguf_path, "rb");
        const size_t base = gguf_get_data_offset(gu);
        auto load = [&](ggml_tensor * t, const char * name) {
            int64_t idx = gguf_find_tensor(gu, name);
            std::vector<uint8_t> buf(ggml_nbytes(t));
            std::fseek(f, (long)(base + gguf_get_tensor_offset(gu, idx)), SEEK_SET);
            if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) { std::printf("read fail\n"); exit(1); }
            ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
        };
        load(k_in, "encoder.conv_in.weight");
        if (b_in) load(b_in, "encoder.conv_in.bias");
        std::fclose(f);
    }

    ggml_init_params gp = { (size_t) 1 << 30, nullptr, true };
    ggml_context * ctx = ggml_init(gp);
    ggml_tensor * x_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, H, W, 1); // torch-labeled
    ggml_tensor * xc = ggml_cont(ctx, ggml_permute(ctx, x_t, 2, 1, 0, 3)); // [W,H,3,B]
    ggml_tensor * y = ggml_conv_2d(ctx, k_in, xc, 1, 1, 1, 1, 1, 1);       // [W,H,OC,1]
    ggml_tensor * y_t = ggml_cont(ctx, ggml_permute(ctx, y, 2, 1, 0, 3));  // torch [OC,H,W]
    ggml_set_input(x_t);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y_t);
    ggml_set_output(y_t);
    ggml_set_output(y); // also read the RAW conv output (w-fastest order)
    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(alloc, gf);
    ggml_backend_tensor_set(x_t, image.data(), 0, image.size() * 4);
    ggml_backend_graph_compute(be, gf);

    std::vector<float> yd((size_t) OC * H * W);
    ggml_backend_tensor_get(y_t, yd.data(), 0, yd.size() * 4);
    std::vector<float> yraw((size_t) OC * H * W);
    ggml_backend_tensor_get(y, yraw.data(), 0, yraw.size() * 4); // [OW,OH,OC,N] flat

    // naive f32 conv on the same data (host), for a few output pixels/channels
    std::vector<float> w((size_t) OC * IC * 9);
    ggml_backend_tensor_get(k_in, w.data(), 0, w.size() * 4);
    // k_in ne [KW,KH,IC,OC] == torch [OC,IC,KH,KW] flat
    auto Wt = [&](int oc, int ic, int kh, int kw) { return w[((size_t) oc * IC + ic) * 9 + kh * 3 + kw]; };
    double worst = 0; int bad = 0;
    const int pts[6][2] = {{0,0},{1,0},{0,1},{5,5},{300,200},{511,511}};
    for (int pt = 0; pt < 6; ++pt)
        for (int oc = 0; oc < 4; ++oc) {
            const int oh = pts[pt][0], ow = pts[pt][1];
            double acc = 0;
            for (int ic = 0; ic < IC; ++ic)
                for (int kh = 0; kh < 3; ++kh)
                    for (int kw = 0; kw < 3; ++kw) {
                        const int ih = oh + kh - 1, iw = ow + kw - 1;
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                        acc += (double) image[((size_t) ic * H + ih) * W + iw] * Wt(oc, ic, kh, kw);
                    }
            const float got = yd[((size_t) oc * H + oh) * W + ow];
            const float got_raw = yraw[(size_t) ow + W * oh + (size_t) W * H * oc];
            const double d = std::fabs(got - acc);
            const double d_raw = std::fabs(got_raw - acc);
            if (d > worst) worst = d;
            if (d > 1e-2) ++bad;
            std::printf("oc=%3d px(%3d,%3d): ggml_t=%10.5f ggml_raw=%10.5f naive=%10.5f d=%.2e d_raw=%.2e\n",
                        oc, oh, ow, got, got_raw, acc, d, d_raw);
        }
    std::printf("%s (worst=%.3e bad=%d)\n", bad ? "FAIL" : "PASS", worst, bad);
    return bad ? 1 : 0;
}
