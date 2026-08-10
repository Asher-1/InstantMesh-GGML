// Manual softmax attention in ggml vs numpy, layout [head_dim, seq, n_head, B].
// This is the reference attention path used by dino.cpp (replaces the buggy
// flash_attn_ext on head_dim=64 in ggml v0.18.0).
//
//   ./build/test_flashattn   -> dumps /tmp/fa_{q,k,v,out}.bin
//   python3 verify the output against an explicit numpy forward.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "ggml.h"
#include "ggml-cpu.h"

static void dump(ggml_backend_t b, ggml_tensor * t, const char * fn) {
    size_t n = ggml_nelements(t);
    std::vector<float> buf(n);
    ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    FILE * f = std::fopen(fn, "wb");
    std::fwrite(buf.data(), sizeof(float), n, f); std::fclose(f);
}

int main() {
    const int hd = 64, nh = 12, seq = 197, B = 1;
    ggml_init_params ip = { 128u<<20, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, nh, B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, nh, B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, nh, B);
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);

    // KQ[kv, q, nh, B] = sum_d k[d,kv,nh]*q[d,q,nh]; softmax over ne0 (kv).
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
    float scale = 1.0f / std::sqrt((float) hd);
    kq = ggml_scale(ctx, kq, scale);
    ggml_tensor * att = ggml_soft_max_ext(ctx, kq, nullptr, 1.0f, 0.0f); // [kv, q, nh, B]
    ggml_tensor * kq_dump = ggml_cont(ctx, kq);

    // o[q, hd, nh] = sum_kv att[kv,q,nh]*v[kv,hd,nh]; mul_mat -> [q, hd, nh, B].
    ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [kv, hd, nh, B]
    ggml_tensor * o = ggml_mul_mat(ctx, att, vp);
    ggml_tensor * o_dump = ggml_cont(ctx, o);
    ggml_tensor * out = ggml_permute(ctx, o, 2, 0, 1, 3); // [hd, nh, q, B]

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, kq_dump);
    ggml_build_forward_expand(gf, o_dump);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (auto * t : {q, k, v}) {
        size_t n = ggml_nelements(t);
        std::vector<float> buf(n);
        for (auto & x : buf) x = ((std::rand() % 1000) / 500.0f) - 1.0f;
        ggml_backend_tensor_set(t, buf.data(), 0, n * sizeof(float));
    }
    ggml_backend_graph_compute(backend, gf);
    dump(backend, q, "/tmp/fa_q.bin");
    dump(backend, k, "/tmp/fa_k.bin");
    dump(backend, v, "/tmp/fa_v.bin");
    dump(backend, att, "/tmp/fa_att.bin");
    dump(backend, kq_dump, "/tmp/fa_kq.bin");
    dump(backend, o_dump, "/tmp/fa_o.bin");
    dump(backend, vp, "/tmp/fa_vp.bin");
    dump(backend, out, "/tmp/fa_out.bin");
    // In-program reference: mul_mat(k,q)[0,0,0] = sum_d k[d,0,0]*q[d,0,0].
    {
        std::vector<float> kb(64), qb(64);
        ggml_backend_tensor_get(k, kb.data(), 0, 64*sizeof(float));
        ggml_backend_tensor_get(q, qb.data(), 0, 64*sizeof(float));
        double s = 0;
        for (int d = 0; d < 64; d++) s += (double)kb[0*64+d] * qb[0*64+d];
        std::printf("prog sum_d k[d,0,0]*q[d,0,0] = %.9g\n", s);
        float kq0;
        ggml_backend_tensor_get(kq_dump, &kq0, 0, sizeof(float));
        std::printf("prog kq_dump[0] = %.9g\n", kq0);
    }
    std::printf("att ne=%lld %lld %lld %lld  out ne=%lld %lld %lld %lld\n",
        (long long)att->ne[0],(long long)att->ne[1],(long long)att->ne[2],(long long)att->ne[3],
        (long long)out->ne[0],(long long)out->ne[1],(long long)out->ne[2],(long long)out->ne[3]);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}