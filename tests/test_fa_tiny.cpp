// Tiny deterministic flash_attn check. q,k,v small, print full output.
#include <cmath>
#include <cstdio>
#include <vector>
#include "ggml.h"
#include "ggml-cpu.h"

int main() {
    const int hd = 2, nh = 1, seq = 3, B = 1;
    ggml_init_params ip = { 16u<<20, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, nh, B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, nh, B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hd, seq, nh, B);
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
    float scale = 1.0f / std::sqrt((float)hd);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_alloc_ctx_tensors(ctx, backend);
    // q[hd,seq]: token0=[1,0], token1=[0,1], token2=[1,1]
    float qd[6] = {1,0, 0,1, 1,1};
    float kd[6] = {1,0, 1,1, 0,1};
    float vd[6] = {1,2, 3,4, 5,6};
    ggml_backend_tensor_set(q, qd, 0, 6*sizeof(float));
    ggml_backend_tensor_set(k, kd, 0, 6*sizeof(float));
    ggml_backend_tensor_set(v, vd, 0, 6*sizeof(float));
    ggml_backend_graph_compute(backend, gf);
    std::vector<float> o(ggml_nelements(out));
    ggml_backend_tensor_get(out, o.data(), 0, o.size()*sizeof(float));
    std::printf("ne=%lld %lld %lld %lld\n",(long long)out->ne[0],(long long)out->ne[1],(long long)out->ne[2],(long long)out->ne[3]);
    for (int t = 0; t < seq; ++t)
        std::printf("token%d out=[%.4f %.4f]\n", t, o[t*2+0], o[t*2+1]);
    return 0;
}