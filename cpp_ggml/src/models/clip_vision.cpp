// Implementation of the zero123++ CLIP vision encoder as a ggml compute graph.
// See clip_vision.hpp for the architecture notes. Graph is rebuilt per forward
// (same pattern as dino.cpp) so any backend works once weights are on it.
#include "models/clip_vision.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-cpu.h"

namespace instantmesh {

namespace {

// layer-0 internal dump points (IM_CV_L0), written after graph compute.
struct DbgEntry { const char * name; ggml_tensor * t; };
static std::vector<DbgEntry> g_dbg;

static void dbg_dump_files() {
    for (auto & e : g_dbg) {
        const size_t n = ggml_nelements(e.t);
        std::vector<float> buf(n);
        ggml_backend_tensor_get(e.t, buf.data(), 0, n * sizeof(float));
        char fn[96]; std::snprintf(fn, sizeof(fn), "/tmp/cv_%s.bin", e.name);
        FILE * f = std::fopen(fn, "wb"); std::fwrite(buf.data(), 4, n, f); std::fclose(f);
    }
}

ggml_tensor * get_t(const ClipVisionModel & m, const char * name) {
    auto it = m.gguf.tensors.find(name);
    if (it == m.gguf.tensors.end()) {
        std::fprintf(stderr, "clip_vision: missing tensor '%s'\n", name);
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

ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x,
                         ggml_tensor * w, ggml_tensor * b, float eps) {
    ggml_tensor * n = ggml_norm(ctx, x, eps);
    n = ggml_mul(ctx, n, w);
    n = ggml_add(ctx, n, b);
    return n;
}

// One CLIPEncoderLayer (post-LN? no — CLIP is pre-LN inside each sublayer with
// a residual add after each; same shape as the ViT layers in dino.cpp).
ggml_tensor * clip_vision_layer(ggml_context * ctx, const ClipVisionModel & m,
                                ggml_tensor * hidden, int i) {
    const ClipVisionHparams & hp = m.hp;
    const int C = hp.hidden_size;
    char name[160];

    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.layer_norm1.weight", i);
    ggml_tensor * ln1w = get_t(m, name);
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.layer_norm1.bias", i);
    ggml_tensor * ln1b = get_t(m, name);
    ggml_tensor * x = layer_norm(ctx, hidden, ln1w, ln1b, hp.layer_norm_eps);

    // self attention: q/k/v without bias (CLIP convention), 16 heads x 80.
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.self_attn.q_proj.weight", i);
    ggml_tensor * wq = get_t(m, name);
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.self_attn.k_proj.weight", i);
    ggml_tensor * wk = get_t(m, name);
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.self_attn.v_proj.weight", i);
    ggml_tensor * wv = get_t(m, name);
    // NOTE: unlike OpenAI's original CLIP, HF's CLIPAttention has biases.
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.self_attn.q_proj.bias", i);
    ggml_tensor * q = ggml_add(ctx, ggml_mul_mat(ctx, wq, x), get_t(m, name));
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.self_attn.k_proj.bias", i);
    ggml_tensor * k = ggml_add(ctx, ggml_mul_mat(ctx, wk, x), get_t(m, name));
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.self_attn.v_proj.bias", i);
    ggml_tensor * v = ggml_add(ctx, ggml_mul_mat(ctx, wv, x), get_t(m, name));
    if (i == 0 && std::getenv("IM_CV_L0")) {
        g_dbg.push_back({"ln1", ggml_cont(ctx, x)});
        g_dbg.push_back({"q", ggml_cont(ctx, q)});
        g_dbg.push_back({"k", ggml_cont(ctx, k)});
        g_dbg.push_back({"v", ggml_cont(ctx, v)});
    }

    const int seq = hidden->ne[1];
    const int B = hidden->ne[2];
    const int hd = C / hp.num_attention_heads;
    const float scale = 1.0f / std::sqrt((float) hd);
    // Manual softmax attention (flash_attn_ext on head_dim=80 is not covered
    // by our hd=64 regression test, and with seq=257 the manual path costs
    // nothing). q/k/v [C, seq, B] -> [hd, seq, nh, B] via c-split (nh outer,
    // hd inner — same order as torch's view(B,seq,nh,hd).transpose).
    ggml_tensor * qr = ggml_permute(ctx, ggml_reshape_4d(ctx, q, hd, hp.num_attention_heads, seq, B), 0, 2, 1, 3);
    ggml_tensor * kr = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k, hd, hp.num_attention_heads, seq, B), 0, 2, 1, 3));
    ggml_tensor * vr = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, hd, hp.num_attention_heads, seq, B), 1, 2, 0, 3)); // [seq_kv, hd, nh, B]
    // KQ[nh, q, seq, B] = sum_hd k[hd,kv]*q[hd,q]; softmax over kv.
    std::fprintf(stderr, "cv: kq a=[%lld %lld %lld %lld] b=[%lld %lld %lld %lld]\n",
        (long long)kr->ne[0], (long long)kr->ne[1], (long long)kr->ne[2], (long long)kr->ne[3],
        (long long)qr->ne[0], (long long)qr->ne[1], (long long)qr->ne[2], (long long)qr->ne[3]);
    ggml_tensor * kq = ggml_mul_mat(ctx, kr, qr); // [seq_kv, seq_q, nh, B]
    kq = ggml_scale(ctx, kq, scale);
    ggml_tensor * att = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f);
    // o[q, hd, nh] = sum_kv att[kv,q]*v[kv,hd]  (v permuted to [kv, hd, nh, B]).
    std::fprintf(stderr, "cv: o a=[%lld %lld %lld %lld] b=[%lld %lld %lld %lld]\n",
        (long long)att->ne[0], (long long)att->ne[1], (long long)att->ne[2], (long long)att->ne[3],
        (long long)vr->ne[0], (long long)vr->ne[1], (long long)vr->ne[2], (long long)vr->ne[3]);
    ggml_tensor * o = ggml_mul_mat(ctx, att, vr); // [seq_q, hd, nh, B]
    // back to [C, seq, B]: permute to [hd, nh, seq] then interleave heads.
    ggml_tensor * attn = ggml_cont(ctx, ggml_permute(ctx, o, 2, 0, 1, 3)); // [hd, nh, seq_q, B]
    attn = ggml_reshape_3d(ctx, attn, C, seq, B);

    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.self_attn.out_proj.weight", i);
    ggml_tensor * wo = get_t(m, name);
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.self_attn.out_proj.bias", i);
    ggml_tensor * bo = get_t(m, name);
    x = linear(ctx, attn, wo, bo);
    if (i == 0 && std::getenv("IM_CV_L0")) {
        g_dbg.push_back({"attn_raw", ggml_cont(ctx, attn)});
        g_dbg.push_back({"attn_proj", ggml_cont(ctx, x)});
    }
    hidden = ggml_add(ctx, hidden, x);

    // MLP: fc1 -> gelu (exact erf, matches HF "gelu") -> fc2.
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.layer_norm2.weight", i);
    ggml_tensor * ln2w = get_t(m, name);
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.layer_norm2.bias", i);
    ggml_tensor * ln2b = get_t(m, name);
    x = layer_norm(ctx, hidden, ln2w, ln2b, hp.layer_norm_eps);
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.mlp.fc1.weight", i);
    ggml_tensor * w1 = get_t(m, name);
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.mlp.fc1.bias", i);
    ggml_tensor * b1 = get_t(m, name);
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.mlp.fc2.weight", i);
    ggml_tensor * w2 = get_t(m, name);
    std::snprintf(name, sizeof(name), "vis.vision_model.encoder.layers.%d.mlp.fc2.bias", i);
    ggml_tensor * b2 = get_t(m, name);
    x = linear(ctx, ggml_gelu_erf(ctx, linear(ctx, x, w1, b1)), w2, b2);
    return ggml_add(ctx, hidden, x);
}

} // namespace

bool clip_vision_load(const std::string & path, ggml_backend_t backend,
                      ClipVisionModel & out, std::string * error) {
    if (!load_gguf(path, backend, out.gguf, error)) return false;
    out.backend = backend;
    const gguf_context * g = out.gguf.gguf;
    out.hp.hidden_size        = kv_i32(g, "vis.hidden_size", 1280);
    out.hp.num_hidden_layers  = kv_i32(g, "vis.num_hidden_layers", 32);
    out.hp.num_attention_heads = kv_i32(g, "vis.num_attention_heads", 16);
    out.hp.intermediate_size  = kv_i32(g, "vis.intermediate_size", 5120);
    out.hp.image_size         = kv_i32(g, "vis.image_size", 224);
    out.hp.patch_size         = kv_i32(g, "vis.patch_size", 14);
    out.hp.projection_dim     = kv_i32(g, "vis.projection_dim", 1024);
    out.hp.layer_norm_eps     = kv_f32(g, "vis.layer_norm_eps", 1e-5f);
    return true;
}

float * clip_vision_encode(const ClipVisionModel & m, const float * image,
                           int B, int * out_dim) {
    const ClipVisionHparams & hp = m.hp;
    const int C = hp.hidden_size;
    const int P = hp.patch_size;
    const int ph = hp.image_size / P;         // 16
    const int seq = 1 + ph * ph;              // 257

    size_t ctx_size = 512u << 20; // patch conv im2col + 32 layers activations
    ggml_init_params ip = { ctx_size, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    // image in ggml layout [W, H, 3, B] — same bytes as torch [B, 3, H, W].
    ggml_tensor * image_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                               hp.image_size, hp.image_size, 3, B);

    // Patch conv (k=P, s=P, no bias). F32 weights take the exact im2col path
    // (ggml_conv_2d quantizes its im2col to F16 — bad for f32 parity).
    ggml_tensor * proj = get_t(m, "vis.vision_model.embeddings.patch_embedding.weight");
    ggml_tensor * conv;
    if (proj->type == GGML_TYPE_F32) {
        ggml_tensor * im2col = ggml_im2col(ctx, proj, image_t, P, P, 0, 0, 1, 1, true, GGML_TYPE_F32);
        ggml_tensor * im2col_2d = ggml_reshape_2d(ctx, im2col, im2col->ne[0],
                                                  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]);
        ggml_tensor * k_2d = ggml_reshape_2d(ctx, proj, proj->ne[0] * proj->ne[1] * proj->ne[2], proj->ne[3]);
        ggml_tensor * r = ggml_mul_mat(ctx, im2col_2d, k_2d);
        r = ggml_cont(ctx, ggml_permute(ctx,
            ggml_reshape_4d(ctx, r, im2col->ne[1], im2col->ne[2], im2col->ne[3], proj->ne[3]),
            0, 1, 3, 2));
        conv = r;
    } else {
        conv = ggml_conv_2d(ctx, proj, image_t, P, P, 0, 0, 1, 1);
    }
    // [OW, OH, OC, N] -> [OC, OW*OH, N] spatial row-major (matches torch
    // flatten(2).transpose(1,2)); same trick as dino.cpp.
    conv = ggml_cont(ctx, ggml_permute(ctx, conv, 1, 2, 0, 3));
    ggml_tensor * patch = ggml_reshape_3d(ctx, conv, C, ph * ph, B);

    // CLS token prepend + position embedding + pre-layernorm.
    ggml_tensor * cls = get_t(m, "vis.vision_model.embeddings.class_embedding");
    cls = ggml_repeat(ctx, cls, ggml_new_tensor_3d(ctx, cls->type, C, 1, B));
    ggml_tensor * hidden = ggml_concat(ctx, cls, patch, 1);
    ggml_tensor * pos = ggml_reshape_3d(ctx, get_t(m, "vis.vision_model.embeddings.position_embedding.weight"),
                                        C, seq, 1);
    hidden = ggml_add(ctx, hidden, pos);
    ggml_tensor * emb_t = hidden;
    hidden = layer_norm(ctx, hidden,
                        get_t(m, "vis.vision_model.pre_layrnorm.weight"),
                        get_t(m, "vis.vision_model.pre_layrnorm.bias"), hp.layer_norm_eps);

    std::vector<ggml_tensor *> layer_outs;
    for (int i = 0; i < hp.num_hidden_layers; ++i) {
        hidden = clip_vision_layer(ctx, m, hidden, i);
        layer_outs.push_back(hidden);
    }

    // Pool the CLS position, then project (no bias) -> [projection_dim, B].
    hidden = layer_norm(ctx, hidden,
                        get_t(m, "vis.vision_model.post_layernorm.weight"),
                        get_t(m, "vis.vision_model.post_layernorm.bias"), hp.layer_norm_eps);
    ggml_tensor * pooled = ggml_view_3d(ctx, hidden, C, 1, B, hidden->nb[1], hidden->nb[2], 0);
    ggml_tensor * result_t = ggml_cont(ctx,
        ggml_mul_mat(ctx, get_t(m, "vis.visual_projection.weight"), pooled)); // [proj, B]

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m.backend));
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result_t);
    const bool dump = std::getenv("IM_CV_DUMP") != nullptr;
    const bool l0 = std::getenv("IM_CV_L0") != nullptr;
    const char * dump_names[6] = {"patch", "emb", "layer0", "layer1", "last", "post_ln"};
    ggml_tensor * dump_src[6] = {patch, emb_t, layer_outs[0], layer_outs[1], layer_outs.back(), hidden};
    ggml_tensor * dump_t[6] = {};
    if (dump) {
        for (int i = 0; i < 6; ++i) {
            dump_t[i] = ggml_cont(ctx, dump_src[i]);
            ggml_build_forward_expand(gf, dump_t[i]);
        }
    }
    if (l0) {
        for (auto & e : g_dbg) ggml_build_forward_expand(gf, e.t);
    }
    ggml_gallocr_alloc_graph(alloc, gf);

    ggml_backend_tensor_set(image_t, image, 0, (size_t) B * 3 * hp.image_size * hp.image_size * sizeof(float));
    ggml_backend_graph_compute(m.backend, gf);

    if (dump) {
        for (int i = 0; i < 6; ++i) {
            const size_t n = ggml_nelements(dump_t[i]);
            std::vector<float> buf(n);
            ggml_backend_tensor_get(dump_t[i], buf.data(), 0, n * sizeof(float));
            char fn[64]; std::snprintf(fn, sizeof(fn), "/tmp/cv_%s.bin", dump_names[i]);
            FILE * f = std::fopen(fn, "wb"); std::fwrite(buf.data(), 4, n, f); std::fclose(f);
        }
    }
    if (l0) dbg_dump_files();

    float * result = (float *) malloc((size_t) B * hp.projection_dim * sizeof(float));
    ggml_backend_tensor_get(result_t, result, 0, (size_t) B * hp.projection_dim * sizeof(float));

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    if (out_dim) *out_dim = hp.projection_dim;
    return result;
}

} // namespace instantmesh
