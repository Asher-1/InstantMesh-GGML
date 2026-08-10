// End-to-end InstantMesh ggml pipeline: multi-view image -> mesh.
//
//   DINO(image, camera) -> image_feats [V,197,768]
//   -> TriplaneTransformer -> planes [1,3,80,64,64]
//   -> OSGDecoder (sampling + sdf/deformation/weight MLPs)
//   -> deformation tanh normalization + empty-shape sdf fix
//   -> v_deformed = grid_verts + deformation
//   -> FlexiCubes isosurface extraction -> triangle mesh
//
// Usage:
//   ./build/instantmesh --dino dino_f16.gguf --transformer transformer_f16.gguf \
//       --synthesizer synthesizer_f16.gguf \
//       --image multiview.bin --camera camera.bin --out mesh.obj
//
// --image  holds [V,3,224,224] float32, RGB, ImageNet-normalized (same as the
//          inputs dino_encode consumes). If omitted a tiny synthetic input is
//          used for a smoke test.
// --camera holds [V,16] float32 (12 extrinsic + 4 intrinsic). If omitted the
//          default 6-view zero123plus camera (radius=4.0, fov=30) is used.
//
// The reference geometry (construct_voxel_grid, sdf fix, deformation
// normalization) mirrors src/models/geometry/rep_3d/flexicubes_geometry.py and
// src/models/lrm_mesh.py.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <tuple>

#include "core/backend.hpp"
#include "models/dino.hpp"
#include "models/lrm_transformer.hpp"
#include "models/synthesizer.hpp"
#include "models/flexicubes.hpp"

// ---------------------------------------------------------------------------
// Small binary helpers
// ---------------------------------------------------------------------------
static bool read_blob(const char * path, std::vector<float> & out) {
    FILE * f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "failed to open %s\n", path); return false; }
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<float> tmp((size_t) n / sizeof(float));
    if (std::fread(tmp.data(), sizeof(float), tmp.size(), f) != tmp.size()) { std::fclose(f); return false; }
    std::fclose(f); out.swap(tmp); return true;
}

// ---------------------------------------------------------------------------
// FlexiCubes voxel grid (mirrors fc.construct_voxel_grid in flexicubes_geometry).
//   cell coords (ix,iy,iz)/res + 8 cube corners -> candidate vertices, then
//   rounded + sorted unique. Returns verts in [-0.5, 0.5]^3 and cube corner
//   indices into verts.
// ---------------------------------------------------------------------------
static void construct_voxel_grid(int res, std::vector<float> & verts,
                                 std::vector<int32_t> & cubes) {
    static const int corners[8][3] = {
        {0,0,0},{1,0,0},{0,1,0},{1,1,0},
        {0,0,1},{1,0,1},{0,1,1},{1,1,1}};
    const long ncell = (long) res * res * res;

    // items: (orig_idx, x, y, z) for all res^3*8 candidate vertices.
    std::vector<std::tuple<long, float, float, float>> items;
    items.reserve(ncell * 8);
    long idx = 0;
    for (int iz = 0; iz < res; ++iz)
    for (int iy = 0; iy < res; ++iy)
    for (int ix = 0; ix < res; ++ix) {
        float bx = (float) ix / res, by = (float) iy / res, bz = (float) iz / res;
        for (int c = 0; c < 8; ++c) {
            float x = std::roundf((bx + corners[c][0] / (float) res) * 1e5f) / 1e5f;
            float y = std::roundf((by + corners[c][1] / (float) res) * 1e5f) / 1e5f;
            float z = std::roundf((bz + corners[c][2] / (float) res) * 1e5f) / 1e5f;
            items.emplace_back(idx, x, y, z);
            ++idx;
        }
    }

    auto before = [](const std::tuple<long,float,float,float> & a,
                     const std::tuple<long,float,float,float> & b) {
        if (std::get<1>(a) != std::get<1>(b)) return std::get<1>(a) < std::get<1>(b);
        if (std::get<2>(a) != std::get<2>(b)) return std::get<2>(a) < std::get<2>(b);
        return std::get<3>(a) < std::get<3>(b);
    };
    std::sort(items.begin(), items.end(), before);

    std::vector<int> remap(items.size());
    std::vector<float> ux, uy, uz;
    int uid = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const auto & it = items[i];
        if (i == 0 || std::get<1>(it) != std::get<1>(items[i-1]) ||
            std::get<2>(it) != std::get<2>(items[i-1]) ||
            std::get<3>(it) != std::get<3>(items[i-1])) {
            ux.push_back(std::get<1>(it));
            uy.push_back(std::get<2>(it));
            uz.push_back(std::get<3>(it));
            ++uid;
        }
        remap[std::get<0>(it)] = uid - 1;
    }

    verts.resize((size_t) uid * 3);
    for (int i = 0; i < uid; ++i) {
        verts[i*3]   = ux[i] - 0.5f;
        verts[i*3+1] = uy[i] - 0.5f;
        verts[i*3+2] = uz[i] - 0.5f;
    }
    cubes.resize(ncell * 8);
    for (long ic = 0; ic < ncell; ++ic)
        for (int c = 0; c < 8; ++c)
            cubes[ic*8 + c] = remap[(size_t) ic*8 + c];
}

// Mirror get_center_boundary_index: indices into the (res+1)^3 vertex grid.
static void center_boundary_indices(int res, std::vector<int> & center,
                                    std::vector<int> & boundary) {
    const int n = res + 1;
    const int c = res / 2 + 1;
    center.clear();
    center.push_back((c * n + c) * n + c); // (c,c,c) linear index

    boundary.clear();
    for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
        if (i < 2 || i > res - 1 || j < 2 || j > res - 1 || k < 2 || k > res - 1)
            boundary.push_back((k * n + j) * n + i);
    }
}

// Homogeneous 3D vector helpers.
struct Vec3 { float x, y, z; };
static Vec3 cross3(const Vec3 & a, const Vec3 & b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
static Vec3 norm3(Vec3 a) {
    float n = std::sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
    if (n > 0) { a.x/=n; a.y/=n; a.z/=n; }
    return a;
}

// Build the default zero123plus input cameras [6,16] (radius=4.0, fov=30).
static std::vector<float> default_cameras() {
    const float radius = 4.0f;
    const float focal = 0.5f / std::tan((float) M_PI / 6.0f * 0.5f); // fov=30
    const float az[6]  = {30, 90, 150, 210, 270, 330};
    const float el[6]  = {20, -10, 20, -10, 20, -10};
    const Vec3 up = {0, 0, 1};
    std::vector<float> cam(6 * 16);
    for (int v = 0; v < 6; ++v) {
        float a = az[v] * (float) M_PI / 180.0f;
        float e = el[v] * (float) M_PI / 180.0f;
        Vec3 pos = {radius * std::cos(e) * std::cos(a),
                    radius * std::cos(e) * std::sin(a),
                    radius * std::sin(e)};
        Vec3 z = norm3(pos);                 // normalize(pos - 0)
        Vec3 x = norm3(cross3(up, z));
        Vec3 y = cross3(z, x);
        float * c = &cam[v * 16];
        // 3x4 extrinsic (row-major: r0.x r0.y r0.z pos.x, ...)
        c[0]=x.x; c[1]=y.x; c[2]=z.x; c[3]=pos.x;
        c[4]=x.y; c[5]=y.y; c[6]=z.y; c[7]=pos.y;
        c[8]=x.z; c[9]=y.z; c[10]=z.z; c[11]=pos.z;
        // intrinsics [fx, fy, cx, cy]
        c[12]=focal; c[13]=focal; c[14]=0.5f; c[15]=0.5f;
    }
    return cam;
}

// ---------------------------------------------------------------------------
static void usage(const char * p) {
    std::fprintf(stderr,
        "usage: %s --dino <dino.gguf> --transformer <transformer.gguf>\n"
        "           --synthesizer <synthesizer.gguf>\n"
        "       [--image multiview.bin] [--camera camera.bin]\n"
        "       [--grid-res N] [--grid-scale S] [--out mesh.obj]\n"
        "       [--dump-sdf sdf.bin] [--device auto|cpu|gpu]\n", p);
}

int main(int argc, char ** argv) {
    const char * dino_path = nullptr, * trans_path = nullptr, * syn_path = nullptr;
    const char * image_path = nullptr, * camera_path = nullptr;
    const char * out_path = "mesh.obj";
    const char * dump_sdf_path = nullptr;
    const char * dump_planes_path = nullptr;
    const char * device = "auto";
    int grid_res = 64;
    float grid_scale = 2.1f;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char * tag, const char ** dst) {
            if (!std::strcmp(argv[i], tag) && i + 1 < argc) *dst = argv[++i];
            else return false;
            return true;
        };
        if (need("--dino", &dino_path) || need("--transformer", &trans_path) ||
            need("--synthesizer", &syn_path) || need("--image", &image_path) ||
            need("--camera", &camera_path) || need("--out", &out_path) ||
            need("--dump-sdf", &dump_sdf_path) ||
            need("--dump-planes", &dump_planes_path) ||
            need("--device", &device)) continue;
        else if (!std::strcmp(argv[i], "--grid-res") && i + 1 < argc) grid_res = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--grid-scale") && i + 1 < argc) grid_scale = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { usage(argv[0]); return 1; }
    }
    if (!dino_path || !trans_path || !syn_path) { usage(argv[0]); return 1; }

    std::string backend_name;
    ggml_backend_t backend = instantmesh::init_best_backend(backend_name, device);
    std::printf("backend: %s\n", backend_name.c_str());

    std::string error;
    instantmesh::DinoModel dino;
    instantmesh::LrmTransformerModel trans;
    instantmesh::SynthesizerModel syn;
    if (!instantmesh::dino_load(dino_path, backend, dino, &error) ||
        !instantmesh::lrm_transformer_load(trans_path, backend, trans, &error) ||
        !instantmesh::synthesizer_load(syn_path, backend, syn, &error)) {
        std::fprintf(stderr, "model load failed: %s\n", error.c_str());
        ggml_backend_free(backend);
        return 1;
    }
    std::printf("dino: hidden=%d layers=%d\n", dino.hp.hidden_size, dino.hp.num_hidden_layers);
    std::printf("transformer: dim=%d layers=%d triplane=%d@%d\n",
        trans.hp.inner_dim, trans.hp.num_layers,
        trans.hp.triplane_dim, trans.hp.triplane_high_res);
    std::printf("synthesizer: plane_dim=%d\n", syn.hp.plane_dim);

    // ---- input: multi-view images [V,3,224,224] + camera [V,16] ----------
    const int IMG = dino.hp.image_size; // 224
    std::vector<float> camera = default_cameras(); // [6,16]
    int V = 6;
    if (camera_path) {
        if (!read_blob(camera_path, camera)) { ggml_backend_free(backend); return 1; }
        if (camera.size() % 16 != 0) {
            std::fprintf(stderr, "camera.bin size %zu not a multiple of 16\n", camera.size());
            ggml_backend_free(backend); return 1;
        }
        V = (int) camera.size() / 16;
    }

    std::vector<float> image((size_t) V * 3 * IMG * IMG);
    if (image_path) {
        std::vector<float> blob;
        if (!read_blob(image_path, blob)) { ggml_backend_free(backend); return 1; }
        if (blob.size() != image.size()) {
            std::fprintf(stderr, "image.bin size mismatch: got %zu, need %zu ([%d,3,%d,%d])\n",
                blob.size(), image.size(), V, IMG, IMG);
            ggml_backend_free(backend); return 1;
        }
        std::memcpy(image.data(), blob.data(), image.size() * sizeof(float));
    } else {
        std::fprintf(stderr, "warning: no --image, using synthetic input\n");
        for (auto & v : image) v = ((float) (std::rand() % 1000) / 1000.0f) - 0.5f;
    }

    // ---- Stage 1: DINO encode (all views at once) ------------------------
    int seq = 0, hidden = 0;
    float * image_feats = instantmesh::dino_encode(dino, image.data(), camera.data(),
                                                   V, IMG, IMG, &seq, &hidden);
    std::printf("dino output: [%d, %d, %d]\n", V, seq, hidden);

    // ---- Stage 2: TriplaneTransformer ------------------------------------
    int n_planes = 0, p_dim = 0, pH = 0, pW = 0;
    float * planes = instantmesh::lrm_transformer_forward(
        trans, image_feats, /*N=*/1, /*n_cond=*/V * seq, &n_planes, &p_dim, &pH, &pW);
    std::printf("planes: [1, %d, %d, %d, %d]\n", n_planes, p_dim, pH, pW);
    std::free(image_feats);

    if (dump_planes_path) {
        FILE * f = std::fopen(dump_planes_path, "wb");
        if (f) {
            std::fwrite(planes, sizeof(float), (size_t) n_planes * p_dim * pH * pW, f);
            std::fclose(f);
            std::printf("dumped planes -> %s (%d floats)\n", dump_planes_path,
                        n_planes * p_dim * pH * pW);
        } else std::fprintf(stderr, "can't write %s\n", dump_planes_path);
    }

    // ---- Stage 3: voxel grid + OSGDecoder -------------------------------
    std::vector<float> verts;
    std::vector<int32_t> cubes;
    construct_voxel_grid(grid_res, verts, cubes);
    const int M = (int) verts.size() / 3;
    const int n_cubes = (int) cubes.size() / 8;
    // scale grid into the object box (mirror parity_synth: verts * grid_scale)
    for (auto & v : verts) v *= grid_scale;
    std::printf("grid: M=%d cubes=%d\n", M, n_cubes);

    float * sdf = nullptr, * deformation = nullptr, * weight = nullptr;
    instantmesh::synthesizer_forward(syn, planes, /*N=*/1, pH, pW,
                                     verts.data(), M, cubes.data(), n_cubes,
                                     &sdf, &deformation, &weight);
    // planes kept alive: needed below to query per-vertex vertex colors.

    // ---- Stage 4: deformation normalization + empty-shape sdf fix --------
    // deformation = 1/(grid_res * 4) * tanh(deformation)
    const float dscale = 1.0f / (grid_res * 4.0f);
    for (int i = 0; i < M * 3; ++i) deformation[i] = std::tanh(deformation[i]) * dscale;

    {
        float mn = sdf[0], mx = sdf[0]; double mean = 0; long pos = 0, neg = 0;
        for (int i = 0; i < M; ++i) { if (sdf[i] < mn) mn = sdf[i]; if (sdf[i] > mx) mx = sdf[i]; mean += sdf[i]; pos += sdf[i] > 0; neg += sdf[i] < 0; }
        std::printf("sdf: [%.4f, %.4f] mean=%.4f pos=%ld neg=%ld\n", mn, mx, mean / M, pos, neg);
    }

    // sdf fix: push a fully-positive / fully-negative volume through the surface.
    {
        // interior vertex sign counts (ignore the 1-voxel boundary band)
        long pos = 0, neg = 0;
        const int n1 = grid_res + 1;
        for (int k = 1; k <= grid_res - 1; ++k)
        for (int j = 1; j <= grid_res - 1; ++j)
        for (int i = 1; i <= grid_res - 1; ++i) {
            float s = sdf[((size_t) k * n1 + j) * n1 + i];
            if (s > 0) ++pos; else if (s < 0) ++neg;
        }
        if (pos == 0 || neg == 0) {
            float mn = sdf[0], mx = sdf[0];
            for (int i = 1; i < M; ++i) { if (sdf[i] < mn) mn = sdf[i]; if (sdf[i] > mx) mx = sdf[i]; }
            std::vector<int> center, boundary;
            center_boundary_indices(grid_res, center, boundary);
            std::vector<float> update(M, 0.0f);
            for (int ci : center)   update[ci] += (1.0f - mn);
            for (int bi : boundary) update[bi] += (-1.0f - mx);
            for (int i = 0; i < M; ++i)
                if (update[i] != 0.0f) sdf[i] = update[i];
            std::fprintf(stderr, "empty-shape sdf fix applied (pos=%ld neg=%ld)\n", pos, neg);
        }
    }

    // ---- Stage 5: FlexiCubes mesh extraction -----------------------------
    if (dump_sdf_path) {
        FILE * f = std::fopen(dump_sdf_path, "wb");
        if (f) {
            std::fwrite(sdf, sizeof(float), (size_t) M, f);
            std::fclose(f);
            std::printf("dumped sdf -> %s (%d floats)\n", dump_sdf_path, M);
        } else std::fprintf(stderr, "can't write %s\n", dump_sdf_path);
    }

    std::vector<float> vdef(M * 3);
    for (int i = 0; i < M * 3; ++i) vdef[i] = verts[i] + deformation[i];

    std::vector<float> beta((size_t) n_cubes * 12), alpha((size_t) n_cubes * 8);
    std::vector<float> gamma((size_t) n_cubes);
    for (int c = 0; c < n_cubes; ++c) {
        for (int k = 0; k < 12; ++k) beta[c*12+k]  = weight[c*21+k];
        for (int k = 0; k < 8;  ++k) alpha[c*8+k]  = weight[c*21+12+k];
        gamma[c] = weight[c*21+20];
    }

    instantmesh::FlexiMesh mesh;
    instantmesh::flexicubes_extract(vdef.data(), sdf, cubes.data(), n_cubes,
                                    beta.data(), alpha.data(), gamma.data(),
                                    grid_res, mesh);
    std::printf("mesh: verts=%zu faces=%zu\n",
                mesh.vertices.size() / 3, mesh.faces.size() / 3);

    // ---- Vertex colors ---------------------------------------------------
    // Sample the triplane at the final mesh vertex positions and push them
    // through net_rgb -> per-vertex RGB in [0,1] (matches upstream InstantMesh
    // get_texture_prediction). Falls back to gray when RGB is unavailable.
    const size_t nv = mesh.vertices.size() / 3;
    std::vector<float> vcolor((size_t) nv * 3, 0.7f);
    float * rgb = nullptr;
    if (instantmesh::synthesizer_texture_forward(syn, planes, 1, pH, pW,
                                                 mesh.vertices.data(), (int) nv,
                                                 &rgb) && rgb) {
        // rgb = sigmoid(x) * (1 + 2*0.001) - 0.001  (MipNeRF clamp, see reference)
        for (size_t i = 0; i < (size_t) nv * 3; ++i) {
            float c = rgb[i] * (1.0f + 2.0f * 0.001f) - 0.001f;
            vcolor[i] = c < 0.0f ? 0.0f : (c > 1.0f ? 1.0f : c);
        }
        std::free(rgb);
        std::printf("vertex color: queried %zu verts\n", nv);
    } else {
        std::fprintf(stderr, "vertex color unavailable (gray fallback)\n");
    }
    std::free(planes);

    // ---- Output OBJ ------------------------------------------------------
    {
        FILE * f = std::fopen(out_path, "w");
        if (!f) { std::fprintf(stderr, "can't write %s\n", out_path); }
        else {
            for (size_t i = 0; i < nv; ++i) {
                const float * p = &mesh.vertices[3 * i];
                const float * c = &vcolor[3 * i];
                std::fprintf(f, "v %g %g %g %g %g %g\n",
                             p[0], p[1], p[2], c[0], c[1], c[2]);
            }
            for (size_t i = 0; i + 2 < mesh.faces.size(); i += 3)
                std::fprintf(f, "f %d %d %d\n", mesh.faces[i]+1, mesh.faces[i+1]+1, mesh.faces[i+2]+1);
            std::fclose(f);
            std::printf("wrote %s (with vertex colors)\n", out_path);
        }
    }

    std::free(sdf); std::free(deformation); std::free(weight);
    ggml_backend_free(backend);
    return 0;
}