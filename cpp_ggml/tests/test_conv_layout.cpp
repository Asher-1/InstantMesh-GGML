// Layout-contract probe: pins down the true memory semantics of ggml v0.21
// conv_2d / group_norm / upscale against in-process naive C++ references
// (no torch, no dumps — the reference loops live in this file).
//
// Established (2026-09-08, this probe):
//   * conv_2d output ne [OW,OH,OC,N] contiguous, memory == torch NCHW —
//     CONFIRMED for asymmetric multi-channel inputs and for chained convs.
//   * conv_2d numerics: internally im2col quantizes windows to F16
//     (ggml.c: a->type == BF16 ? F32 : F16), so parity vs an f32 reference
//     floors at ~f16 epsilon (relative) — a precision property, NOT a
//     layout/numerics bug.
//   * group_norm on contiguous [W,H,C,N] == torch GroupNorm semantics
//     (biased variance, groups = consecutive channels) — exact.
//   * upscale(NEAREST, 2) scales ne0/ne1 of [W,H,C,N] — exact.
//
// The conv sweep also covers VAE-realistic shapes (IC=3, OC=128) to catch
// shape-dependent CPU kernel regressions.
#include <cmath>
#include <cstdio>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"

namespace {

uint32_t lcg_state = 12345;
float frand() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return ((lcg_state >> 8) & 0xffff) / 16384.0f - 2.0f; // ~[-2,2)
}

struct Fails {
    int n = 0;
    double maxd = 0.0;
    double rel = 5e-3; // relative tolerance; chained convs accumulate the f16 floor
    // relative threshold: layout errors produce O(1) scrambles, while
    // the f16 im2col floor scales with magnitude (~f16 eps per sum).
    void check(const char * what, double got, double exp, size_t idx) {
        const double d = std::fabs(got - exp);
        if (d > maxd) maxd = d;
        if (!(d < rel * (1.0 + std::fabs(exp)))) {
            if (n < 5)
                std::printf("MISMATCH %s idx=%zu got=%.6f exp=%.6f\n", what, idx, got, exp);
            ++n;
        }
    }
};

// Naive NCHW conv2d reference. in flat: torch [N,IC,H,W]; w flat: ggml
// kernel ne [KW,KH,IC,OC] (== torch [OC,IC,KH,KW] contiguous); out flat:
// torch [N,OC,OH,OW].
std::vector<float> naive_conv(const std::vector<float> & in, int N, int IC,
                              int H, int W, const std::vector<float> & w,
                              const std::vector<float> & bias, int OC,
                              int KH, int KW, int s, int p) {
    const int OH = (H + 2 * p - (KH - 1) - 1) / s + 1;
    const int OW = (W + 2 * p - (KW - 1) - 1) / s + 1;
    std::vector<float> out((size_t) N * OC * OH * OW, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int oc = 0; oc < OC; ++oc)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    double acc = bias.empty() ? 0.0 : bias[oc];
                    for (int ic = 0; ic < IC; ++ic)
                        for (int kh = 0; kh < KH; ++kh)
                            for (int kw = 0; kw < KW; ++kw) {
                                const int ih = oh * s + kh - p;
                                const int iw = ow * s + kw - p;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                acc += (double) in[((size_t) n * IC + ic) * H * W +
                                                   (size_t) ih * W + iw] *
                                       w[((size_t) oc * IC + ic) * KH * KW +
                                         (size_t) kh * KW + kw];
                            }
                    out[((size_t) n * OC + oc) * OH * OW + (size_t) oh * OW + ow] = (float) acc;
                }
    return out;
}

int run_conv_case(int W, int H, int IC, int OC, int N, bool with_bias) {
    const int K = 3, S = 1, P = 1;
    const int OH = (H + 2 * P - K) / S + 1, OW = (W + 2 * P - K) / S + 1;
    ggml_init_params ip = { (size_t) 1 << 30, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, IC, N);
    ggml_tensor * wk = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, K, IC, OC);
    ggml_set_input(x); ggml_set_input(wk);
    // NOTE: ggml_conv_2d has NO bias parameter — bias is a manual ggml_add
    // with a [1,1,OC,1] broadcast view (exactly what vae.cpp does).
    ggml_tensor * y = ggml_conv_2d(ctx, wk, x, S, S, P, P, 1, 1);
    ggml_tensor * b = nullptr;
    std::vector<float> bd;
    if (with_bias) {
        b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC);
        ggml_set_input(b);
        y = ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, OC, 1));
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_alloc_ctx_tensors(ctx, be);

    std::vector<float> xd(ggml_nelements(x)), wd(ggml_nelements(wk));
    for (auto & v : xd) v = frand();
    for (auto & v : wd) v = frand() * 0.1f; // conv-like |w| ~ 0.1
    if (with_bias) { bd.resize(OC); for (auto & v : bd) v = frand(); }
    ggml_backend_tensor_set(x, xd.data(), 0, xd.size() * 4);
    ggml_backend_tensor_set(wk, wd.data(), 0, wd.size() * 4);
    if (with_bias) ggml_backend_tensor_set(b, bd.data(), 0, bd.size() * 4);
    ggml_backend_graph_compute(be, gf);

    Fails f;
    if (!ggml_is_contiguous(y)) { std::printf("conv_2d out NOT contiguous\n"); ++f.n; }
    if (y->ne[0] != OW || y->ne[1] != OH || y->ne[2] != OC || y->ne[3] != N) {
        std::printf("conv_2d out ne = [%lld %lld %lld %lld] exp [%d %d %d %d]\n",
                    (long long) y->ne[0], (long long) y->ne[1],
                    (long long) y->ne[2], (long long) y->ne[3], OW, OH, OC, N);
        ++f.n;
    }
    // flat memory == naive torch NCHW [N,OC,OH,OW]
    std::vector<float> yd(ggml_nelements(y));
    ggml_backend_tensor_get(y, yd.data(), 0, yd.size() * 4);
    const std::vector<float> ref = naive_conv(xd, N, IC, H, W, wd, bd, OC, K, K, S, P);
    for (int n = 0; n < N; ++n)
        for (int oc = 0; oc < OC; ++oc)
            for (int ih = 0; ih < OH; ++ih)
                for (int iw = 0; iw < OW; ++iw)
                    f.check("conv_2d[NCHW]",
                            yd[(size_t) iw + OW * ih + (size_t) OW * OH * oc +
                               (size_t) OW * OH * OC * n],
                            ref[((size_t) n * OC + oc) * OH * OW + (size_t) ih * OW + iw],
                            (size_t) n * OC * OH * OW + (size_t) oc * OH * OW +
                                (size_t) ih * OW + iw);
    std::printf("%s conv_2d %dx%dx%d->%d n=%d %s (mismatches=%d max_abs=%.2e)\n",
                f.n ? "FAIL" : "PASS", W, H, IC, OC, N, with_bias ? "bias" : "nobias",
                f.n, f.maxd);
    ggml_backend_free(be);
    ggml_free(ctx);
    return f.n;
}

} // namespace

int main() {
    int fails = 0;

    // ── conv_2d sweep: layout + numerics across shapes ───────────────────
    // small asymmetric, VAE-realistic channels, larger spatial, stride-2.
    fails += run_conv_case(7, 5, 3, 4, 2, true);
    fails += run_conv_case(16, 16, 3, 128, 1, false); // encoder.conv_in_raw shape class
    fails += run_conv_case(24, 18, 3, 128, 1, true);
    fails += run_conv_case(64, 48, 3, 128, 1, false);
    fails += run_conv_case(16, 16, 128, 128, 1, true); // resnet-scale
    fails += run_conv_case(512, 512, 3, 128, 1, false); // encoder.conv_in exact shape

    // ── conv_2d chain: feed output straight into the next conv ──────────
    {
        const int W = 6, H = 4, C0 = 3, C1 = 5, C2 = 2, N = 1, K = 3, S = 1, P = 1;
        ggml_init_params ip = { (size_t) 64 << 20, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C0, N);
        ggml_tensor * w1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, K, C0, C1);
        ggml_tensor * b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C1);
        ggml_tensor * w2 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, K, C1, C2);
        ggml_tensor * b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C2);
        ggml_set_input(x); ggml_set_input(w1); ggml_set_input(b1);
        ggml_set_input(w2); ggml_set_input(b2);
        ggml_tensor * y1 = ggml_conv_2d(ctx, w1, x, S, S, P, P, 1, 1); // [W,H,C1,N]
        y1 = ggml_add(ctx, y1, ggml_reshape_4d(ctx, b1, 1, 1, C1, 1));
        ggml_tensor * y2 = ggml_conv_2d(ctx, w2, y1, S, S, P, P, 1, 1); // consumes y1 directly
        y2 = ggml_add(ctx, y2, ggml_reshape_4d(ctx, b2, 1, 1, C2, 1));
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y2);
        ggml_backend_t be = ggml_backend_cpu_init();
        ggml_backend_alloc_ctx_tensors(ctx, be);
        std::vector<float> xd(ggml_nelements(x)), w1d(ggml_nelements(w1)), b1d(C1),
            w2d(ggml_nelements(w2)), b2d(C2);
        for (auto & v : xd) v = frand();
        for (auto & v : w1d) v = frand();
        for (auto & v : b1d) v = frand();
        for (auto & v : w2d) v = frand();
        for (auto & v : b2d) v = frand();
        ggml_backend_tensor_set(x, xd.data(), 0, xd.size() * 4);
        ggml_backend_tensor_set(w1, w1d.data(), 0, w1d.size() * 4);
        ggml_backend_tensor_set(b1, b1d.data(), 0, b1d.size() * 4);
        ggml_backend_tensor_set(w2, w2d.data(), 0, w2d.size() * 4);
        ggml_backend_tensor_set(b2, b2d.data(), 0, b2d.size() * 4);
        ggml_backend_graph_compute(be, gf);
        std::vector<float> y2d(ggml_nelements(y2));
        ggml_backend_tensor_get(y2, y2d.data(), 0, y2d.size() * 4);
        const std::vector<float> r1 = naive_conv(xd, N, C0, H, W, w1d, b1d, C1, K, K, S, P);
        const std::vector<float> r2 = naive_conv(r1, N, C1, H, W, w2d, b2d, C2, K, K, S, P);
        Fails f;
        f.rel = 2e-2; // two convs: the f16 window floor adds up (~1e-2 rel)
        for (size_t i = 0; i < r2.size(); ++i) f.check("conv_chain", y2d[i], r2[i], i);
        std::printf("%s conv_2d chained without reorder (mismatches=%d max_abs=%.2e)\n",
                    f.n ? "FAIL" : "PASS", f.n, f.maxd);
        fails += f.n;
        ggml_backend_free(be);
        ggml_free(ctx);
    }

    // ── group_norm on contiguous [W,H,C,N] (torch semantics) ────────────
    {
        const int W = 5, H = 4, C = 8, N = 2, G = 4;
        const float eps = 1e-6f;
        ggml_init_params ip = { (size_t) 64 << 20, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, N);
        ggml_set_input(x);
        ggml_tensor * y = ggml_group_norm(ctx, x, G, eps);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_t be = ggml_backend_cpu_init();
        ggml_backend_alloc_ctx_tensors(ctx, be);
        std::vector<float> xd(ggml_nelements(x));
        for (auto & v : xd) v = frand();
        ggml_backend_tensor_set(x, xd.data(), 0, xd.size() * 4);
        ggml_backend_graph_compute(be, gf);
        std::vector<float> yd(ggml_nelements(y));
        ggml_backend_tensor_get(y, yd.data(), 0, yd.size() * 4);
        // naive: per (n, group) mean/var over channels-in-group × spatial
        Fails f;
        const int cpg = C / G;
        for (int n = 0; n < N; ++n)
            for (int g = 0; g < G; ++g) {
                double sum = 0, sum2 = 0;
                const size_t cnt = (size_t) cpg * H * W;
                for (int c = g * cpg; c < (g + 1) * cpg; ++c)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w) {
                            const float v = xd[(size_t) w + W * h + (size_t) W * H * c +
                                               (size_t) W * H * C * n];
                            sum += v; sum2 += (double) v * v;
                        }
                const float mean = (float) (sum / cnt);
                const float var = (float) (sum2 / cnt - (double) mean * mean);
                for (int c = g * cpg; c < (g + 1) * cpg; ++c)
                    for (int h = 0; h < H; ++h)
                        for (int w = 0; w < W; ++w) {
                            const size_t idx = (size_t) w + W * h + (size_t) W * H * c +
                                               (size_t) W * H * C * n;
                            const float expv = (xd[idx] - mean) / std::sqrt(var + eps);
                            f.check("group_norm", yd[idx], expv, idx);
                        }
            }
        std::printf("%s group_norm on [W,H,C,N] contiguous (mismatches=%d)\n",
                    f.n ? "FAIL" : "PASS", f.n);
        fails += f.n;
        ggml_backend_free(be);
        ggml_free(ctx);
    }

    // ── upscale nearest ×2 on [W,H,C,N] ──────────────────────────────────
    {
        const int W = 4, H = 3, C = 2, N = 1, F = 2;
        ggml_init_params ip = { (size_t) 64 << 20, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, N);
        ggml_set_input(x);
        ggml_tensor * y = ggml_upscale(ctx, x, F, GGML_SCALE_MODE_NEAREST);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_t be = ggml_backend_cpu_init();
        ggml_backend_alloc_ctx_tensors(ctx, be);
        std::vector<float> xd(ggml_nelements(x));
        for (auto & v : xd) v = frand();
        ggml_backend_tensor_set(x, xd.data(), 0, xd.size() * 4);
        ggml_backend_graph_compute(be, gf);
        std::vector<float> yd(ggml_nelements(y));
        ggml_backend_tensor_get(y, yd.data(), 0, yd.size() * 4);
        Fails f;
        if (y->ne[0] != W * F || y->ne[1] != H * F || y->ne[2] != C || y->ne[3] != N) {
            std::printf("upscale ne = [%lld %lld %lld %lld] exp [%d %d %d %d]\n",
                        (long long) y->ne[0], (long long) y->ne[1],
                        (long long) y->ne[2], (long long) y->ne[3], W * F, H * F, C, N);
            ++f.n;
        }
        for (int c = 0; c < C; ++c)
            for (int h = 0; h < H * F; ++h)
                for (int w = 0; w < W * F; ++w) {
                    const size_t idx = (size_t) w + W * F * h + (size_t) W * F * H * F * c;
                    const float expv = xd[(size_t) (w / F) + W * (h / F) + (size_t) W * H * c];
                    f.check("upscale", yd[idx], expv, idx);
                }
        std::printf("%s upscale nearest x2 on [W,H,C,N] (mismatches=%d)\n",
                    f.n ? "FAIL" : "PASS", f.n);
        fails += f.n;
        ggml_backend_free(be);
        ggml_free(ctx);
    }

    std::printf("%s\n", fails ? "FAILURES PRESENT" : "ALL PASS");
    return fails ? 1 : 0;
}
