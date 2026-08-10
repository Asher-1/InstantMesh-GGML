// Implementation of the geometry-prediction (OSGDecoder) as a ggml graph.
//
// Two parts:
//  1. Host triplane sampling: the fixed bilinear grid_sample that maps grid
//     vertices onto the 3 planes (no weights, not a ggml graph). This is the
//     memory-heavy step (sampled [3C, M] where M = (grid_res+1)^3).
//  2. The OSGDecoder MLPs (sdf / deformation / weight) as a single ggml graph
//     scheduled on the selected backend, so those weights run on CPU/CUDA/
//     Vulkan identically.
#include "models/synthesizer.hpp"
#include <chrono>
#include <cstdlib>

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#ifdef IM_USE_OPENMP
#include <omp.h>
#endif

#include "ggml-alloc.h"
#include "ggml-cpu.h"

namespace instantmesh {

namespace {

ggml_tensor * get_t(const SynthesizerModel & m, const char * name) {
    std::string key = std::string("decoder.") + name;
    auto it = m.gguf.tensors.find(key);
    if (it == m.gguf.tensors.end()) {
        std::fprintf(stderr, "synthesizer: missing tensor '%s'\n", key.c_str());
        std::abort();
    }
    return it->second;
}

ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x,
                     ggml_tensor * W, ggml_tensor * b) {
    ggml_tensor * y = ggml_mul_mat(ctx, W, x);
    if (b) y = ggml_add(ctx, y, b);
    return y;
}

// One OSGDecoder MLP: Linear(3C->64) ReLU (64->64) ReLU ... ReLU Linear(64->out).
// Weight/bias indexed 2*i (net_sdf.0/.2/.4/.6). Input x is [fin, n], output
// [fout, n].
ggml_tensor * mlp(ggml_context * ctx, const SynthesizerModel & m,
                  ggml_tensor * x, const char * tag) {
    const SynthesizerHparams & hp = m.hp;
    char name[128];
    for (int i = 0; i < hp.num_layers; ++i) {
        std::snprintf(name, sizeof(name), "%s.%d.weight", tag, 2 * i);
        ggml_tensor * W = get_t(m, name);
        std::snprintf(name, sizeof(name), "%s.%d.bias", tag, 2 * i);
        ggml_tensor * b = get_t(m, name);
        x = linear(ctx, x, W, b);
        if (i < hp.num_layers - 1) x = ggml_relu(ctx, x);
    }
    return x;
}

// Bilinear grid sample (align_corners=False, padding zeros) of one plane.
// plane: [C, H, W] contiguous; nx,ny normalized in [-1,1]. Writes out[c].
// Maps nx -> px = (nx+1)/2*W - 0.5, then 4-neighbour weighted sum.
static void sample_plane(const float * plane, int C, int H, int W,
                         float nx, float ny, float * out) {
    const float px = (nx + 1.0f) * 0.5f * W - 0.5f;
    const float py = (ny + 1.0f) * 0.5f * H - 0.5f;
    const int x0 = (int) std::floor(px);
    const int y0 = (int) std::floor(py);
    const float fx = px - (float) x0;
    const float fy = py - (float) y0;
    // weights of the 4 neighbours (x0/x0+1, y0/y0+1).
    const float w[4] = { (1 - fx) * (1 - fy), fx * (1 - fy),
                         (1 - fx) * fy, fx * fy };
    const int xs[4] = { x0, x0 + 1, x0, x0 + 1 };
    const int ys[4] = { y0, y0, y0 + 1, y0 + 1 };
    for (int c = 0; c < C; ++c) {
        float acc = 0.0f;
        for (int k = 0; k < 4; ++k) {
            if (xs[k] >= 0 && xs[k] < W && ys[k] >= 0 && ys[k] < H) {
                acc += w[k] * plane[c * H * W + ys[k] * W + xs[k]];
            }
        }
        out[c] = acc;
    }
}

// Host triplane sampling. planes [N,3,C,H,W]; verts [M,3]; writes sampled
// [N, 3C, M] (row = plane*C + channel). For N>1 the same verts are used for
// every batch element.
void sample_triplanes(const SynthesizerModel & m,
                      const float * planes, int N, int H, int W,
                      const float * verts, int M, float * sampled) {
    const int C = m.hp.plane_dim;
    const int plane_stride = C * H * W;
    const int total = N * M;
#ifdef IM_USE_OPENMP
    #pragma omp parallel for schedule(static) num_threads(std::max(1, omp_get_max_threads()))
#endif
    for (int it = 0; it < total; ++it) {
        const int n = it / M;
        const int v = it % M;
        float tmp[3 * 128];  // plane_dim <= 128
        const float * pb = planes + (size_t) n * 3 * plane_stride;
        const float * p = verts + 3 * v;
        const float x = p[0], y = p[1], z = p[2];
        // plane 0 : (x,y) ; plane 1 : (x,z) ; plane 2 : (z,y)
        sample_plane(pb + 0 * plane_stride, C, H, W, x, y, &tmp[0 * C]);
        sample_plane(pb + 1 * plane_stride, C, H, W, x, z, &tmp[1 * C]);
        sample_plane(pb + 2 * plane_stride, C, H, W, z, y, &tmp[2 * C]);
        std::memcpy(sampled + (size_t) it * (3 * C), tmp, (size_t) 3 * C * sizeof(float));
    }
}

} // namespace

bool synthesizer_load(const std::string & path, ggml_backend_t backend,
                      SynthesizerModel & out, std::string * error) {
    out.backend = backend;
    if (!load_gguf(path, backend, out.gguf, error)) return false;
    const gguf_context * g = out.gguf.gguf;
    out.hp.plane_dim  = (int) kv_i32(g, "synthesizer.plane_dim", 80);
    out.hp.hidden     = (int) kv_i32(g, "synthesizer.hidden", 64);
    out.hp.num_layers = (int) kv_i32(g, "synthesizer.num_layers", 4);
    return true;
}

void synthesizer_forward(const SynthesizerModel & model,
                         const float * planes, int N, int H, int W,
                         const float * verts, int M,
                         const int32_t * cubes, int n_cubes,
                         float ** outsdf, float ** outdeform, float ** outweight) {
    const bool prof = std::getenv("SYNTH_PROFILE") != nullptr;
    double t0 = prof ? now_s() : 0;
    const SynthesizerHparams & hp = model.hp;
    const int F = 3 * hp.plane_dim;                 // per-vertex feature width
    const int GF = 8 * F;                           // per-cube feature width

    // ---- host geometry (fixed ops, no weights) ---------------------------
    std::vector<float> sampled((size_t) N * M * F);
    sample_triplanes(model, planes, N, H, W, verts, M, sampled.data());
    double t_samp = prof ? now_s() : 0;

    size_t ns = (size_t) N * M;
    size_t nd = (size_t) N * M * 3;
    size_t nw = (size_t) N * n_cubes * 21;
    float * S = (float *) malloc(ns * sizeof(float));
    float * D = (float *) malloc(nd * sizeof(float));
    float * Wt = (float *) malloc(nw * sizeof(float));

    // ---- ggml graph for the OSGDecoder MLPs ------------------------------
    // net_sdf / net_deformation run once on the sampled triplane features
    // [F, M*N]. F32 activations: the FlexiCubes SDF/deformation/weight outputs
    // are too precision-sensitive for F16 (quantizing them to F16 corrupts the
    // extracted mesh — it balloons from ~39k to ~1.8M faces), so geometry MLPs
    // stay F32.
    ggml_init_params iparams = { /*mem_size=*/16u << 20, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(iparams);

    ggml_tensor * sampled_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, M * N);
    ggml_set_input(sampled_t);

    ggml_tensor * sdf        = mlp(ctx, model, sampled_t, "net_sdf");         // [1, M*N]
    ggml_tensor * deformation = mlp(ctx, model, sampled_t, "net_deformation"); // [3, M*N]

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sdf);
    ggml_build_forward_expand(gf, deformation);
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(sampled_t, sampled.data(), 0, sampled.size() * sizeof(float));
    ggml_backend_graph_compute(model.backend, gf);
    ggml_backend_tensor_get(sdf, S, 0, ns * sizeof(float));
    ggml_backend_tensor_get(deformation, D, 0, nd * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    double t_sdf = prof ? now_s() : 0;

    // ---- net_weight (FlexiCubes): grid [8F, n_cubes] built on the GPU with
    // ggml_get_rows(sampled, cubes) -> [F, 8*nc] then a corner-major
    // reshape/permute/cont into [8F, n_cubes]. This keeps the multi-GB
    // intermediate on the GPU (avoids a ~3s CPU gather + multi-GB host->device
    // copy). Because that intermediate is large, the cubes are processed in
    // chunks to bound GPU memory. ---- 
    // Chunk size trades graph-setup overhead (favours large chunks on GPU) against
    // peak intermediate memory (get_rows + cont buffers). Larger chunks cut the
    // per-chunk ggml context/gallocr/graph overhead ~2x (1<<16 -> 1<<18), but on
    // the CPU backend big contiguous gathers are slower, so keep a small chunk there.
    const int chunk = ggml_backend_is_cpu(model.backend) ? (1 << 16) : (1 << 18);
    for (int n = 0; n < N; ++n) {
        const int32_t off = n * M;
        for (int base = 0; base < n_cubes; base += chunk) {
            const int cn = std::min(chunk, n_cubes - base);
            std::vector<int32_t> idx((size_t) 8 * cn);
            for (int c = 0; c < cn; ++c)
                for (int v = 0; v < 8; ++v)
                    idx[c * 8 + v] = cubes[(base + c) * 8 + v] + off;

            ggml_init_params ip2 = { /*mem_size=*/16u << 20, nullptr, /*no_alloc=*/true };
            ggml_context * c2 = ggml_init(ip2);
            ggml_tensor * st = ggml_new_tensor_2d(c2, GGML_TYPE_F32, F, M * N);
            ggml_set_input(st);
            ggml_tensor * b = ggml_new_tensor_1d(c2, GGML_TYPE_I32, (int64_t) 8 * cn);
            ggml_set_input(b);
            ggml_tensor * g = ggml_get_rows(c2, st, b);              // [F, 8*cn]
            g = ggml_reshape_3d(c2, g, F, 8, cn);                    // [F,8,cn]
            g = ggml_permute(c2, g, 1, 0, 2, 3);                     // [8,F,cn]
            g = ggml_cont_2d(c2, g, GF, cn);                         // [8F,cn]
            ggml_tensor * w = mlp(c2, model, g, "net_weight");       // [21, cn]
            w = ggml_scale(c2, w, 0.1f);

            ggml_gallocr_t a2 = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
            ggml_cgraph * g2 = ggml_new_graph(c2);
            ggml_build_forward_expand(g2, w);
            ggml_gallocr_alloc_graph(a2, g2);

            ggml_backend_tensor_set(st, sampled.data(), 0, sampled.size() * sizeof(float));
            ggml_backend_tensor_set(b, idx.data(), 0, idx.size() * sizeof(int32_t));
            ggml_backend_graph_compute(model.backend, g2);

            float * wcol = Wt + ((size_t) n * n_cubes + base) * 21;
            ggml_backend_tensor_get(w, wcol, 0, (size_t) 21 * cn * sizeof(float));

            ggml_gallocr_free(a2);
            ggml_free(c2);
        }
    }
    if (prof) std::printf("SYNTH_PROFILE fwd: sample=%.3fs sdf/deform=%.3fs weight=%.3fs (M=%d ncubes=%d)\n",
                          t_samp - t0, t_sdf - t_samp, now_s() - t_sdf, M, n_cubes);

    *outsdf = S;
    *outdeform = D;
    *outweight = Wt;
}

bool synthesizer_texture_forward(const SynthesizerModel & model,
                                 const float * planes, int N, int H, int W,
                                 const float * points, int M, float ** rgb) {
    const bool prof = std::getenv("SYNTH_PROFILE") != nullptr;
    double t0 = prof ? now_s() : 0;
    const SynthesizerHparams & hp = model.hp;
    const int F = 3 * hp.plane_dim;

    // Sample the triplane at the query points, then run net_rgb.
    std::vector<float> sampled((size_t) N * M * F);
    sample_triplanes(model, planes, N, H, W, points, M, sampled.data());
    double t_host = prof ? now_s() : 0;

    ggml_init_params iparams = { /*mem_size=*/8u << 20, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(iparams);

    ggml_tensor * sampled_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, M * N);
    ggml_set_input(sampled_t);

    ggml_tensor * rgb_t = mlp(ctx, model, sampled_t, "net_rgb");   // [3, M*N]
    rgb_t = ggml_sigmoid(ctx, rgb_t);  // affine clamp applied on host

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rgb_t);
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(sampled_t, sampled.data(), 0, sampled.size() * sizeof(float));
    ggml_backend_graph_compute(model.backend, gf);
    if (prof) std::printf("SYNTH_PROFILE tex: host=%.3fs ggml=%.3fs (M=%d)\n",
                          t_host - t0, now_s() - t_host, M);

    size_t n = (size_t) N * M * 3;
    float * R = (float *) malloc(n * sizeof(float));
    ggml_backend_tensor_get(rgb_t, R, 0, n * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    *rgb = R;
    return true;
}

} // namespace instantmesh