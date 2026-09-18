// zero123pp — Zero123++ multi-view diffusion CLI (ggml port).
//
// Replicates the official `Zero123PlusPipeline.__call__` path used by run.py:
//
//   input image
//    ├─ feature_extractor_vae (resize 512, crop 512, mean .5 / std .8)
//    │      ─→ VAE.encode(mode) ─→ cond_lat [1,4,64,64]
//    │      cfg branch: negative = precomputed zero-image latent (GGUF)
//    ├─ feature_extractor_clip (resize 224, crop 224, CLIP mean/std)
//    │      ─→ CLIPVisionModelWithProjection ─→ global_embeds [1,1024]
//    ├─ context [B,77,1024] = cat(empty-text, empty-text + global_embeds*ramp)
//    └─ 75-step EulerAncestral (trailing, v_prediction, guidance 4.0) with
//         RefOnly double-forward per step (w-pass 64x64 cond, r-pass 120x80)
//      ─→ latents [1,4,120,80] → unscale_latents → VAE.decode → unscale_image
//      ─→ 960x640 3x2 grid → view_0..5 (320x320) + grid.png
//
// The per-step random inputs (initial latents, cond noise, ancestral noise)
// come from an internal deterministic RNG by default; pass --fixture-dir to
// consume noise tensors dumped by convert/dump_e2e.py for exact parity runs.
//
//   usage: zero123pp --image in.png --out DIR [--cond c.gguf] [--unet u.gguf]
//                     [--vae v.gguf] [--steps 75] [--seed 42] [--guidance 4.0]
//                     [--device auto|cpu] [--fixture-dir DIR]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <filesystem>
#include <chrono>

#include "core/backend.hpp"
#include "models/unet.hpp"
#include "models/vae.hpp"
#include "models/clip_vision.hpp"
#include "models/scheduler.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "utils/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "utils/stb_image_write.h"

// ---------------------------------------------------------------------------
// tiny IO helpers
// ---------------------------------------------------------------------------
static bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    std::FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t) n);
    if (n > 0 && std::fread(out.data(), 1, (size_t) n, f) != (size_t) n) { std::fclose(f); return false; }
    std::fclose(f);
    return true;
}

static bool read_blob(const std::string & path, std::vector<float> & out, size_t expect) {
    std::vector<uint8_t> raw;
    if (!read_file(path, raw)) return false;
    if (raw.size() != expect * sizeof(float)) return false;
    out.resize(expect);
    std::memcpy(out.data(), raw.data(), raw.size());
    return true;
}

static bool write_blob(const std::string & path, const float * data, size_t n) {
    std::FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fwrite(data, sizeof(float), n, f);
    std::fclose(f);
    return true;
}

static void read_backend_tensor(ggml_backend_t backend, const ggml_tensor * t,
                                std::vector<float> & out) {
    out.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

static double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// deterministic RNG (standalone mode; torch CPU float32 Box-Muller shape)
// ---------------------------------------------------------------------------
struct Rng {
    std::mt19937 g;
    explicit Rng(uint32_t seed) : g(seed) {}
    float uniform() {
        // generate_canonical over 24 bits: uniform in [0, 1)
        return std::generate_canonical<float, 24>(g);
    }
    void randn(float * out, size_t n) {
        for (size_t k = 0; k < n; k += 2) {
            const float u1 = uniform(), u2 = uniform();
            const float r = std::sqrt(-2.f * std::log(1.f - u1));
            const float th = 2.f * 3.14159265358979323846f * u2;
            out[k] = r * std::cos(th);
            if (k + 1 < n) out[k + 1] = r * std::sin(th);
        }
    }
};

// ---------------------------------------------------------------------------
// image preprocessing (feature_extractor_vae / feature_extractor_clip)
// ---------------------------------------------------------------------------
// Catmull-Rom cubic, matches PIL's BICUBIC resample kernel closely.
static float cubic_w(float x) {
    const float a = -0.5f;
    x = std::fabs(x);
    if (x < 1.f) return (a + 2.f) * x * x * x - (a + 3.f) * x * x + 1.f;
    if (x < 2.f) return a * x * x * x - 5.f * a * x * x + 8.f * a * x - 4.f * a;
    return 0.f;
}

// src: [sh, sw, sc] row-major float; dst: [dh, dw, sc].
static void bicubic_resize(const std::vector<float> & src, int sh, int sw, int sc,
                           std::vector<float> & dst, int dh, int dw) {
    dst.assign((size_t) dh * dw * sc, 0.f);
    const float sx = (float) sw / (float) dw;
    const float sy = (float) sh / (float) dh;
    for (int y = 0; y < dh; ++y) {
        const float fy = (y + 0.5f) * sy - 0.5f;
        int iy = (int) std::floor(fy);
        const float ty = fy - iy;
        float wy[4];
        for (int j = -1; j <= 2; ++j) wy[j + 1] = cubic_w(ty - j);
        for (int x = 0; x < dw; ++x) {
            const float fx = (x + 0.5f) * sx - 0.5f;
            int ix = (int) std::floor(fx);
            const float tx = fx - ix;
            float wx[4];
            for (int j = -1; j <= 2; ++j) wx[j + 1] = cubic_w(tx - j);
            for (int c = 0; c < sc; ++c) {
                float acc = 0.f, wn = 0.f;
                for (int j = 0; j < 4; ++j) {
                    const int yy = iy + j - 1;
                    if (yy < 0 || yy >= sh) continue;
                    for (int i = 0; i < 4; ++i) {
                        const int xx = ix + i - 1;
                        if (xx < 0 || xx >= sw) continue;
                        const float w = wy[j] * wx[i];
                        acc += w * src[((size_t) yy * sw + xx) * sc + c];
                        wn += w;
                    }
                }
                dst[((size_t) y * dw + x) * sc + c] = (wn != 0.f) ? acc / wn : 0.f;
            }
        }
    }
}

// src: [sh, sw, sc]; center-crop [ch, cw] into dst.
static void center_crop(const std::vector<float> & src, int sh, int sw, int sc,
                        int ch, int cw, std::vector<float> & dst) {
    const int y0 = (sh - ch) / 2, x0 = (sw - cw) / 2;
    dst.assign((size_t) ch * cw * sc, 0.f);
    for (int y = 0; y < ch; ++y)
        for (int x = 0; x < cw; ++x)
            for (int c = 0; c < sc; ++c)
                dst[((size_t) y * cw + x) * sc + c] =
                    src[((size_t) (y0 + y) * sw + (x0 + x)) * sc + c];
}

// Resize shortest edge to `target`, center-crop `target`x`target`, then
// normalize with per-channel (mean, std) — mirroring transformers
// CLIPImageProcessor (resize → center crop → /255 → (x-mean)/std).
static void preprocess_image(const uint8_t * rgb, int H, int W,
                             int target, float mean, const float * std,
                             std::vector<float> & out) {
    const int sc = 3;
    std::vector<float> src((size_t) H * W * sc);
    for (size_t k = 0; k < src.size(); ++k) src[k] = rgb[k];
    const float scale = (float) target / (float) (H < W ? H : W);
    const int nh = (int) std::lround(H * scale), nw = (int) std::lround(W * scale);
    std::vector<float> rs;
    bicubic_resize(src, H, W, sc, rs, nh, nw);
    std::vector<float> cr;
    center_crop(rs, nh, nw, sc, target, target, cr);
    out.resize((size_t) target * target * sc);
    for (int y = 0; y < target; ++y)
        for (int x = 0; x < target; ++x)
            for (int c = 0; c < sc; ++c)
                out[((size_t) y * target + x) * sc + c] =
                    (cr[((size_t) y * target + x) * sc + c] / 255.f - mean) / std[c];
}

// ---------------------------------------------------------------------------
static void usage(const char * p) {
    std::fprintf(stderr,
        "usage: %s --image <in.png|jpg> --out <dir>\n"
        "       [--cond cond.gguf] [--unet unet.gguf] [--vae vae.gguf]\n"
        "       [--steps 75] [--seed 42] [--guidance 4.0] [--device auto|cpu]\n"
        "       [--fixture-dir DIR]\n", p);
}

int main(int argc, char ** argv) {
    const char * image_path = nullptr, * out_dir = nullptr;
    std::string cond_path = "models/gguf/zero123pp_cond_f32.gguf";
    std::string unet_path = "models/gguf/zero123pp_unet_f16.gguf";
    std::string vae_path  = "models/gguf/zero123pp_vae_f16.gguf";
    const char * fixture_dir = nullptr;
    const char * dump_lat = nullptr;
    const char * dump_steps = nullptr;
    const char * device = "auto";
    int steps = 75;
    uint32_t seed = 42;
    float guidance = 4.0f;

    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char * n) { return std::strcmp(argv[i], n) == 0 && i + 1 < argc; };
        if      (arg("--image"))       image_path = argv[++i];
        else if (arg("--out"))         out_dir    = argv[++i];
        else if (arg("--cond"))        cond_path  = argv[++i];
        else if (arg("--unet"))        unet_path  = argv[++i];
        else if (arg("--vae"))         vae_path   = argv[++i];
        else if (arg("--fixture-dir")) fixture_dir = argv[++i];
        else if (arg("--dump-final-latents")) dump_lat = argv[++i];
        else if (arg("--dump-steps")) dump_steps = argv[++i];
        else if (arg("--device"))      device     = argv[++i];
        else if (arg("--steps"))       steps      = std::atoi(argv[++i]);
        else if (arg("--seed"))        seed       = (uint32_t) std::atoi(argv[++i]);
        else if (arg("--guidance"))    guidance   = (float) std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (!image_path || !out_dir) { usage(argv[0]); return 1; }
    if (steps < 2 || steps > 1000) { std::fprintf(stderr, "--steps out of range\n"); return 1; }

    std::string backend_name;
    ggml_backend_t backend = instantmesh::init_best_backend(backend_name, device);
    std::printf("backend: %s\n", backend_name.c_str());

    // ── load models ────────────────────────────────────────────────────────
    instantmesh::UnetModel unet;
    instantmesh::VaeModel vae;
    instantmesh::ClipVisionModel clip;
    std::string err;
    if (!instantmesh::unet_load(unet_path, backend, unet, &err) ||
        !instantmesh::vae_load(vae_path, backend, vae, &err) ||
        !instantmesh::clip_vision_load(cond_path, backend, clip, &err)) {
        std::fprintf(stderr, "model load failed: %s\n", err.c_str());
        ggml_backend_free(backend);
        return 1;
    }

    // ── input image (RGBA composited over white = official to_rgb_image) ──
    std::vector<uint8_t> img_bytes;
    if (!read_file(image_path, img_bytes)) { std::fprintf(stderr, "cannot read %s\n", image_path); return 1; }
    int iw = 0, ih = 0, ic = 0;
    uint8_t * px = stbi_load_from_memory(img_bytes.data(), (int) img_bytes.size(), &iw, &ih, &ic, 4);
    if (!px) { std::fprintf(stderr, "stbi decode failed: %s\n", image_path); return 1; }
    std::vector<uint8_t> rgb((size_t) ih * iw * 3);
    for (int y = 0; y < ih; ++y)
        for (int x = 0; x < iw; ++x) {
            const uint8_t * p = px + (size_t) (y * iw + x) * 4;
            const float a = p[3] / 255.f;
            for (int c = 0; c < 3; ++c)
                rgb[((size_t) y * iw + x) * 3 + c] = (uint8_t) (p[c] * a + 255.f * (1.f - a));
        }
    stbi_image_free(px);
    std::printf("input: %dx%d\n", iw, ih);

    // ── pipeline constants from the cond GGUF ──────────────────────────────
    const int crop_vae  = (int) instantmesh::kv_i32(clip.gguf.gguf, "fx_vae.crop", 512);
    const int crop_clip = (int) instantmesh::kv_i32(clip.gguf.gguf, "fx_clip.crop", 224);
    std::vector<float> vae_std  = {0.8f, 0.8f, 0.8f};
    const float clip_mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    const float clip_std[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

    std::vector<float> text_emb, neg_lat, alphas;
    auto tt = clip.gguf.tensors.find("cond.text_emb");
    auto tn = clip.gguf.tensors.find("cond.negative_lat");
    auto ta = clip.gguf.tensors.find("scheduler.alphas_cumprod");
    if (tt == clip.gguf.tensors.end() || tn == clip.gguf.tensors.end() ||
        ta == clip.gguf.tensors.end()) {
        std::fprintf(stderr, "cond GGUF missing pipeline constants\n");
        return 1;
    }
    read_backend_tensor(backend, tt->second, text_emb);
    read_backend_tensor(backend, tn->second, neg_lat);
    read_backend_tensor(backend, ta->second, alphas);
    const int L = 77;
    std::vector<float> ramp(L);
    {
        const char * rs = instantmesh::kv_str(clip.gguf.gguf, "cond.ramp", "");
        int n = 0;
        const char * p = rs;
        while (p && *p && n < L) { ramp[n++] = (float) std::atof(p); p = std::strchr(p, ','); if (p) ++p; }
        if (n != L) { std::fprintf(stderr, "cond.ramp parse failed (n=%d)\n", n); return 1; }
    }

    // ── preprocess: VAE branch (crop_vae) and CLIP branch (224) ────────────
    std::vector<float> img_vae, img_clip;
    preprocess_image(rgb.data(), ih, iw, crop_vae, 0.5f, vae_std.data(), img_vae);
    preprocess_image(rgb.data(), ih, iw, crop_clip, 0.f, clip_std, img_clip); // mean applied below
    // CLIP mean is per-channel, not scalar: redo the normalize step directly.
    {
        const int C = crop_clip;
        std::vector<float> raw = img_clip; // currently (x/255 - 0)/std
        img_clip.resize((size_t) C * C * 3);
        for (int y = 0; y < C; ++y)
            for (int x = 0; x < C; ++x)
                for (int c = 0; c < 3; ++c)
                    img_clip[((size_t) y * C + x) * 3 + c] =
                        raw[((size_t) y * C + x) * 3 + c] - clip_mean[c] / clip_std[c];
    }
    if (fixture_dir) {
        // Optional strict-parity overrides for the preprocessed inputs.
        std::vector<float> fi;
        if (read_blob(std::string(fixture_dir) + "/img_vae.bin", fi, (size_t) crop_vae * crop_vae * 3)) {
            img_vae.swap(fi);
            std::printf("fixture: img_vae.bin override\n");
        }
        if (read_blob(std::string(fixture_dir) + "/img_clip.bin", fi, (size_t) crop_clip * crop_clip * 3)) {
            img_clip.swap(fi);
            std::printf("fixture: img_clip.bin override\n");
        }
    }

    // ── condition latent [B,4,64,64]: negative = zero-image latent (GGUF) ──
    const int B = guidance > 1.f ? 2 : 1;
    int lh = 0, lw = 0;
    float * pos_lat = instantmesh::vae_encode_mode(vae, img_vae.data(), 1, crop_vae, crop_vae, &lh, &lw);
    if (!pos_lat) { std::fprintf(stderr, "vae_encode_mode failed\n"); return 1; }
    const size_t lat64 = (size_t) 4 * 64 * 64;
    std::vector<float> cond_lat((size_t) B * lat64);
    std::vector<float> pos_buf(lat64), neg_buf(lat64);
    const float * pos_ptr = pos_lat, * neg_ptr = neg_lat.data();
    if (fixture_dir) {
        // Optional strict-parity overrides for the VAE-encoded condition latents.
        const std::string d = fixture_dir;
        if (read_blob(d + "/cond_pos.bin", pos_buf, lat64)) { pos_ptr = pos_buf.data(); std::printf("fixture: cond_pos.bin override\n"); }
        if (read_blob(d + "/cond_neg.bin", neg_buf, lat64)) { neg_ptr = neg_buf.data(); std::printf("fixture: cond_neg.bin override\n"); }
    }
    if (B == 2) std::memcpy(cond_lat.data(), neg_ptr, lat64 * sizeof(float));
    std::memcpy(cond_lat.data() + (size_t) (B - 1) * lat64, pos_ptr, lat64 * sizeof(float));
    free(pos_lat);

    // ── CLIP vision embeds + context [B,77,1024] ───────────────────────────
    int emb_dim = 0;
    float * glob = instantmesh::clip_vision_encode(clip, img_clip.data(), 1, &emb_dim);
    if (!glob || emb_dim != 1024) { std::fprintf(stderr, "clip_vision_encode failed\n"); return 1; }
    std::vector<float> context((size_t) B * L * 1024);
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < L; ++t)
            for (int d = 0; d < 1024; ++d) {
                float v = text_emb[(size_t) t * 1024 + d];
                if (b == B - 1) v += glob[d] * ramp[t];
                context[((size_t) b * L + t) * 1024 + d] = v;
            }
    free(glob);

    // ── scheduler (config from cond GGUF KV) ───────────────────────────────
    EulerAncestralScheduler::Config scfg;
    scfg.num_train_timesteps = (int) instantmesh::kv_i32(clip.gguf.gguf, "sched.num_train_timesteps", 1000);
    scfg.beta_start          = instantmesh::kv_f32(clip.gguf.gguf, "sched.beta_start", 0.00085f);
    scfg.beta_end            = instantmesh::kv_f32(clip.gguf.gguf, "sched.beta_end", 0.012f);
    scfg.beta_schedule       = instantmesh::kv_str(clip.gguf.gguf, "sched.beta_schedule", "linear");
    scfg.prediction_type     = instantmesh::kv_str(clip.gguf.gguf, "sched.prediction_type", "v_prediction");
    scfg.timestep_spacing    = instantmesh::kv_str(clip.gguf.gguf, "sched.timestep_spacing", "trailing");
    EulerAncestralScheduler sched(scfg);
    sched.set_alphas_cumprod(alphas.data(), (int) alphas.size());
    sched.set_timesteps(steps);

    // ── denoising loop ─────────────────────────────────────────────────────
    const int H = 120, W = 80;                 // latents from height=960, width=640
    const int ref_h = 64, ref_w = 64;          // condition latent
    const size_t lat_n = (size_t) B * 4 * H * W / B;   // per-batch element count
    const size_t lat_total = (size_t) B * 4 * H * W;
    std::vector<float> latents(lat_n);
    Rng rng(seed);

    if (fixture_dir) {
        std::string d = fixture_dir;
        if (!read_blob(d + "/latents0.bin", latents, lat_n)) {
            std::fprintf(stderr, "fixture missing: %s/latents0.bin\n", d.c_str()); return 1;
        }
    } else {
        rng.randn(latents.data(), lat_n);
    }
    const float init_sigma = sched.sigma(0);
    for (size_t k = 0; k < lat_n; ++k) latents[k] *= init_sigma;

    std::vector<float> scaled_in(lat_total), cond_noise((size_t) B * lat64),
                       noisy_cond((size_t) B * lat64), scaled_cond((size_t) B * lat64),
                       noise(lat_n), eps_buf(lat_total);

    const double t0 = now_s();
    for (int i = 0; i < steps; ++i) {
        const float t = sched.timesteps()[i];
        // r-pass input: cat([latents, latents]) then scale_model_input
        std::memcpy(scaled_in.data(), latents.data(), lat_n * sizeof(float));
        std::memcpy(scaled_in.data() + lat_n, latents.data(), lat_n * sizeof(float));
        sched.scale_model_input(scaled_in.data(), (int) lat_total, scaled_in.data());

        // w-pass input: add_noise(cond_lat, cond_noise, t) then scale_model_input
        if (fixture_dir) {
            char fn[256];
            std::snprintf(fn, sizeof(fn), "%s/cond_noise_%03d.bin", fixture_dir, i);
            if (!read_blob(fn, cond_noise, (size_t) B * lat64)) {
                std::fprintf(stderr, "fixture missing: %s\n", fn); return 1;
            }
        } else {
            rng.randn(cond_noise.data(), (size_t) B * lat64);
        }
        sched.add_noise(cond_lat.data(), cond_noise.data(), (int) (B * lat64), t, noisy_cond.data());
        sched.scale_model_input(noisy_cond.data(), (int) (B * lat64), scaled_cond.data());

        if (dump_steps) {
            char fn[256];
            std::snprintf(fn, sizeof(fn), "%s/r_input_%03d.bin", dump_steps, i);
            if (FILE * f = std::fopen(fn, "wb")) { std::fwrite(scaled_in.data(), sizeof(float), lat_total, f); std::fclose(f); }
            std::snprintf(fn, sizeof(fn), "%s/w_input_%03d.bin", dump_steps, i);
            if (FILE * f = std::fopen(fn, "wb")) { std::fwrite(scaled_cond.data(), sizeof(float), (size_t) B * lat64, f); std::fclose(f); }
            std::snprintf(fn, sizeof(fn), "%s/context_%03d.bin", dump_steps, i);
            if (FILE * f = std::fopen(fn, "wb")) { std::fwrite(context.data(), sizeof(float), (size_t) B * L * 1024, f); std::fclose(f); }
        }

        if (fixture_dir) {
            char fn[256];
            std::snprintf(fn, sizeof(fn), "%s/context_%03d.bin", fixture_dir, i);
            std::vector<float> fc;
            if (read_blob(fn, fc, (size_t) B * L * 1024)) context = std::move(fc);
        }

        int oh = 0, ow = 0;
        float * eps = instantmesh::unet_forward_refonly(unet, scaled_in.data(), scaled_cond.data(),
                                                        context.data(), L, B, H, W, ref_h, ref_w,
                                                        t, &oh, &ow);
        if (!eps) { std::fprintf(stderr, "unet forward failed at step %d\n", i); return 1; }
        if (dump_steps) {
            char fn[256];
            std::snprintf(fn, sizeof(fn), "%s/eps_%03d.bin", dump_steps, i);
            if (FILE * f = std::fopen(fn, "wb")) { std::fwrite(eps, sizeof(float), lat_total, f); std::fclose(f); }
        }
        // classifier-free guidance
        for (size_t k = 0; k < lat_n; ++k) {
            eps_buf[k] = eps[k] + guidance * (eps[k + lat_n] - eps[k]);
        }
        free(eps);

        // ancestral noise + scheduler step
        if (fixture_dir) {
            char fn[256];
            std::snprintf(fn, sizeof(fn), "%s/step_noise_%03d.bin", fixture_dir, i);
            if (!read_blob(fn, noise, lat_n)) {
                std::fprintf(stderr, "fixture missing: %s\n", fn); return 1;
            }
        } else {
            rng.randn(noise.data(), lat_n);
        }
        sched.step(eps_buf.data(), latents.data(), noise.data(), (int) lat_n, latents.data());

        // post-step latents (matches dump_e2e latents_%03d.bin = step OUTPUT)
        if (dump_steps) {
            char fn[256];
            std::snprintf(fn, sizeof(fn), "%s/latents_%03d.bin", dump_steps, i);
            if (FILE * f = std::fopen(fn, "wb")) { std::fwrite(latents.data(), sizeof(float), lat_n, f); std::fclose(f); }
        }

        if (i == 0 || (i + 1) % 10 == 0 || i == steps - 1)
            std::printf("step %3d/%d  t=%7.2f  (%.1fs elapsed)\n",
                        i + 1, steps, t, now_s() - t0);
    }

    // ── postprocess: unscale_latents → VAE.decode → unscale_image → u8 ─────
    if (dump_lat) {
        FILE * f = std::fopen(dump_lat, "wb");
        if (!f) { std::fprintf(stderr, "failed to open %s\n", dump_lat); return 1; }
        std::fwrite(latents.data(), sizeof(float), lat_n, f);
        std::fclose(f);
        std::printf("wrote final latents to %s\n", dump_lat);
    }
    for (size_t k = 0; k < lat_n; ++k) latents[k] = latents[k] / 0.75f + 0.22f;

    int dh = 0, dw = 0;
    float * dec = instantmesh::vae_decode(vae, latents.data(), 1, H, W, &dh, &dw);
    if (!dec) { std::fprintf(stderr, "vae_decode failed\n"); return 1; }
    const int OH = dh, OW = dw;                       // 960 x 640
    std::vector<uint8_t> out_rgb((size_t) OH * OW * 3);
    for (size_t k = 0; k < (size_t) OH * OW * 3; ++k) {
        float v = dec[k] / 0.5f * 0.8f;               // unscale_image
        v = v / 2.f + 0.5f;                           // postprocess denormalize
        v = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
        out_rgb[k] = (uint8_t) (v * 255.f + 0.5f);
    }
    free(dec);

    // ── split 3x2 grid into 6 views (run.py rearrange 'c (n h) (m w)') ────
    const int V = 320, rows = 3, cols = 2;
    std::string od = out_dir;
    if (od.back() != '/') od += '/';
    std::error_code ec;
    std::filesystem::create_directories(od, ec);
    const std::string grid_path = od + "grid.png";
    if (!stbi_write_png(grid_path.c_str(), OW, OH, 3, out_rgb.data(), OW * 3))
        std::fprintf(stderr, "failed to write %s\n", grid_path.c_str());
    else
        std::printf("wrote %s (%dx%d)\n", grid_path.c_str(), OW, OH);

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const int idx = r * cols + c;
            std::vector<uint8_t> view((size_t) V * V * 3);
            for (int y = 0; y < V; ++y)
                for (int x = 0; x < V; ++x)
                    for (int ch = 0; ch < 3; ++ch)
                        view[((size_t) y * V + x) * 3 + ch] =
                            out_rgb[((size_t) (r * V + y) * OW + (c * V + x)) * 3 + ch];
            char fn[256];
            std::snprintf(fn, sizeof(fn), "%sview_%d.png", od.c_str(), idx);
            if (!stbi_write_png(fn, V, V, 3, view.data(), V * 3))
                std::fprintf(stderr, "failed to write %s\n", fn);
            else
                std::printf("wrote %s\n", fn);
        }

    std::printf("done in %.1fs\n", now_s() - t0);
    ggml_backend_free(backend);
    return 0;
}
