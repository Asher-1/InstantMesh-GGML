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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
    std::vector<float> tmp(3 * C);
    for (int n = 0; n < N; ++n) {
        const float * pb = planes + (size_t) n * 3 * plane_stride;
        for (int v = 0; v < M; ++v) {
            const float * p = verts + 3 * v;
            const float x = p[0], y = p[1], z = p[2];
            // plane 0 : (x,y) ; plane 1 : (x,z) ; plane 2 : (z,y)
            sample_plane(pb + 0 * plane_stride, C, H, W, x, y, &tmp[0 * C]);
            sample_plane(pb + 1 * plane_stride, C, H, W, x, z, &tmp[1 * C]);
            sample_plane(pb + 2 * plane_stride, C, H, W, z, y, &tmp[2 * C]);
            float * col = sampled + ((size_t) n * M + v) * (3 * C);
            std::memcpy(col, tmp.data(), (size_t) 3 * C * sizeof(float));
        }
    }
}

// Host gather of the 8 cube-corner features per cube. sampled [N,3C,M],
// cubes [n_cubes,8]; writes grid [N, 8*3C, n_cubes] (row = v*3C + f).
void gather_cube_features(const SynthesizerModel & m,
                          const float * sampled, int N, int M,
                          const int32_t * cubes, int n_cubes,
                          float * grid) {
    const int F = 3 * m.hp.plane_dim;
    for (int n = 0; n < N; ++n) {
        const float * sp = sampled + (size_t) n * M * F;
        float * g = grid + (size_t) n * n_cubes * (8 * F);
        for (int c = 0; c < n_cubes; ++c) {
            float * col = g + (size_t) c * (8 * F);
            for (int v = 0; v < 8; ++v) {
                const int vi = cubes[c * 8 + v];
                std::memcpy(col + (size_t) v * F, sp + (size_t) vi * F,
                            (size_t) F * sizeof(float));
            }
        }
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
    const SynthesizerHparams & hp = model.hp;
    const int F = 3 * hp.plane_dim;                 // per-vertex feature width
    const int GF = 8 * F;                           // per-cube feature width

    // ---- host geometry (fixed ops, no weights) ---------------------------
    std::vector<float> sampled((size_t) N * M * F);
    sample_triplanes(model, planes, N, H, W, verts, M, sampled.data());
    std::vector<float> grid((size_t) N * n_cubes * GF);
    gather_cube_features(model, sampled.data(), N, M, cubes, n_cubes, grid.data());

    // ---- ggml graph for the OSGDecoder MLPs ------------------------------
    ggml_init_params iparams = { /*mem_size=*/16u << 20, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(iparams);

    ggml_tensor * sampled_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, M * N);
    ggml_set_input(sampled_t);
    ggml_tensor * grid_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GF, n_cubes * N);
    ggml_set_input(grid_t);

    ggml_tensor * sdf        = mlp(ctx, model, sampled_t, "net_sdf");         // [1, M*N]
    ggml_tensor * deformation = mlp(ctx, model, sampled_t, "net_deformation"); // [3, M*N]
    ggml_tensor * weight_t   = mlp(ctx, model, grid_t, "net_weight");          // [21, n_cubes*N]
    weight_t = ggml_scale(ctx, weight_t, 0.1f);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, sdf);
    ggml_build_forward_expand(gf, deformation);
    ggml_build_forward_expand(gf, weight_t);
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(sampled_t, sampled.data(), 0, sampled.size() * sizeof(float));
    ggml_backend_tensor_set(grid_t, grid.data(), 0, grid.size() * sizeof(float));
    ggml_backend_graph_compute(model.backend, gf);

    size_t ns = (size_t) N * M;
    size_t nd = (size_t) N * M * 3;
    size_t nw = (size_t) N * n_cubes * 21;
    float * S = (float *) malloc(ns * sizeof(float));
    float * D = (float *) malloc(nd * sizeof(float));
    float * Wt = (float *) malloc(nw * sizeof(float));
    ggml_backend_tensor_get(sdf, S, 0, ns * sizeof(float));
    ggml_backend_tensor_get(deformation, D, 0, nd * sizeof(float));
    ggml_backend_tensor_get(weight_t, Wt, 0, nw * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    *outsdf = S;
    *outdeform = D;
    *outweight = Wt;
}

bool synthesizer_texture_forward(const SynthesizerModel & model,
                                 const float * planes, int N, int H, int W,
                                 const float * points, int M, float ** rgb) {
    const SynthesizerHparams & hp = model.hp;
    const int F = 3 * hp.plane_dim;

    // Sample the triplane at the query points, then run net_rgb.
    std::vector<float> sampled((size_t) N * M * F);
    sample_triplanes(model, planes, N, H, W, points, M, sampled.data());

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

    size_t n = (size_t) N * M * 3;
    float * R = (float *) malloc(n * sizeof(float));
    ggml_backend_tensor_get(rgb_t, R, 0, n * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);

    *rgb = R;
    return true;
}

} // namespace instantmesh