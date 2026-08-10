// Minimal deterministic mul_mat to nail ggml semantics exactly.
// k,q as 4D [hd, seq, nh=1, B]. Use tiny hd=2, seq=2 so we can hand-check.
#include <cstdio>
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
    const int hd = 2, seq = 2, nh = 2, B = 1;
    ggml_init_params ip = { 64u<<20, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, nh, B);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, nh, B);
    ggml_set_input(k); ggml_set_input(q);

    // k[hd,seq,nh]: k[d,s,h] = d*10+s + h*100  (head0: 0,10,1,11 ; head1: +100)
    // q[hd,seq,nh]: q[d,s,h] = h*1000 + d*100 + s*10000
    const int n = hd*seq*nh;
    std::vector<float> kd(n), qd(n);
    for (int h = 0; h < nh; h++)
        for (int s = 0; s < seq; s++)
            for (int d = 0; d < hd; d++) {
                kd[s*hd + d + h*hd*seq] = d*10 + s + h*100;
                qd[s*hd + d + h*hd*seq] = h*1000 + d*100 + s*10000;
            }
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
    ggml_tensor * kqc = ggml_cont(ctx, kq);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, kqc);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(k, kd.data(), 0, n*sizeof(float));
    ggml_backend_tensor_set(q, qd.data(), 0, n*sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    std::printf("kq ne=%lld %lld %lld %lld\n",
        (long long)kq->ne[0],(long long)kq->ne[1],(long long)kq->ne[2],(long long)kq->ne[3]);
    std::vector<float> buf(n);
    ggml_backend_tensor_get(kqc, buf.data(), 0, n*sizeof(float));
    std::printf("flat kq =");
    for (auto x : buf) std::printf(" %g", x);
    std::printf("\n");
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}