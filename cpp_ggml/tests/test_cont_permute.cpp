// Verify ggml_cont(permute(reshape_4d(src, hd, nh, seq), 0,2,1,3)) produces a
// [hd, seq, nh] tensor whose flat memory equals torch q[seq, head, hd] flat.
// src is [hidden=hd*nh, seq] with flat = r + c*hidden (r = head*hd + hd).
// Desired out[hd_d, seq_s, nh_h] = src_flat[seq_s*hidden + nh_h*hd + hd_d].
#include <cstdio>
#include <vector>
#include "ggml.h"
#include "ggml-cpu.h"

static void dump(ggml_backend_t b, ggml_tensor * t, const char * fn) {
    size_t n = ggml_nelements(t);
    std::vector<float> buf(n);
    ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    FILE * f = std::fopen(fn, "wb"); std::fwrite(buf.data(), sizeof(float), n, f); std::fclose(f);
}

int main() {
    const int hd=2, nh=3, seq=5, hidden=hd*nh;
    ggml_init_params ip = { 64u<<20, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, seq, 1);
    ggml_set_input(src);
    // fill src so that src_flat[r + c*hidden] = 1000*r + c  (r = head*hd+hd)
    ggml_tensor * t = ggml_reshape_4d(ctx, src, hd, nh, seq, 1); // [hd,nh,seq,1]
    t = ggml_permute(ctx, t, 0,2,1,3);                            // [hd,seq,nh,1]
    t = ggml_cont(ctx, t);                                       // contiguous [hd,seq,nh,1]
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t);
    ggml_backend_t b = ggml_backend_cpu_init();
    ggml_backend_alloc_ctx_tensors(ctx, b);
    {
        size_t n = ggml_nelements(src);
        std::vector<float> buf(n);
        for (int i=0;i<(int)n;i++) buf[i] = 1000.0f*i;
        ggml_backend_tensor_set(src, buf.data(), 0, n*sizeof(float));
    }
    ggml_backend_graph_compute(b, gf);
    dump(b, t, "/tmp/cp_out.bin");
    // expected: out[hd_d, seq_s, nh_h] = src_flat[seq_s*6 + nh_h*2 + hd_d]
    //          = 1000*(seq_s*6 + nh_h*2 + hd_d) + 0
    std::vector<float> ob(ggml_nelements(t));
    ggml_backend_tensor_get(t, ob.data(), 0, ob.size()*sizeof(float));
    int err=0;
    for (int hd_d=0;hd_d<hd;hd_d++)for(int seq_s=0;seq_s<seq;seq_s++)for(int nh_h=0;nh_h<nh;nh_h++){
        int idx = hd_d + seq_s*hd + nh_h*hd*seq;
        float expv = 1000.0f*(seq_s*hidden + nh_h*hd + hd_d);
        if (ob[idx] != expv) { if(err<5) printf("MISMATCH [%d,%d,%d] got %g exp %g\n",hd_d,seq_s,nh_h,ob[idx],expv); err++; }
    }
    printf("cont(permute(reshape)) mismatches: %d\n", err);
    ggml_backend_free(b);
    return err?1:0;
}