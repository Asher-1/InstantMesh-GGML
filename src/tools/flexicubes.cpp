// FlexiCubes isosurface mesh extraction example.
// Reads the synthesizer outputs (verts/cubes/sdf/deformation/weight), applies
// the deformation, and extracts a triangle mesh. Dumps vertices & faces.
//
//   ./build/flexicubes --in-dir /tmp/synth_ref --grid-res 64 --out-dir /tmp/mesh_cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "models/flexicubes.hpp"

static std::vector<float> readF(const std::string & dir, const char * name) {
    std::string p = dir + "/" + name;
    FILE * f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "can't open %s\n", p.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<float> v(sz / sizeof(float));
    if (sz) std::fread(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
    return v;
}
static std::vector<int32_t> readI(const std::string & dir, const char * name) {
    std::string p = dir + "/" + name;
    FILE * f = std::fopen(p.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "can't open %s\n", p.c_str()); std::exit(1); }
    std::fseek(f, 0, SEEK_END); long sz = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v(sz / sizeof(int32_t));
    if (sz) std::fread(v.data(), sizeof(int32_t), v.size(), f);
    std::fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    std::string in_dir = "/tmp/synth_ref", out_dir = "/tmp/mesh_cpp";
    int grid_res = 64;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--in-dir") && i + 1 < argc) in_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--out-dir") && i + 1 < argc) out_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--grid-res") && i + 1 < argc) grid_res = std::atoi(argv[++i]);
    }
    auto verts = readF(in_dir, "verts.bin");
    auto cubes = readI(in_dir, "cubes.bin");
    auto sdf = readF(in_dir, "sdf.bin");
    auto deform = readF(in_dir, "deformation.bin");
    auto weight = readF(in_dir, "weight.bin");

    const int M = (int) verts.size() / 3;
    const int C = (int) cubes.size() / 8;
    std::fprintf(stderr, "verts=%d cubes=%d sdf=%zu deform=%zu weight=%zu\n",
                 M, C, sdf.size(), deform.size(), weight.size());

    // v_deformed = verts + deformation
    std::vector<float> vdef(M * 3);
    for (int i = 0; i < M * 3; ++i) vdef[i] = verts[i] + deform[i];
    std::fprintf(stderr, "vdef[0] = %g %g %g  (verts[0]=%g %g %g def[0]=%g %g %g)\n",
                 vdef[0],vdef[1],vdef[2], verts[0],verts[1],verts[2], deform[0],deform[1],deform[2]);

    // weight: [C, 21] -> beta [C,12], alpha [C,8], gamma [C]
    std::vector<float> beta(C*12), alpha(C*8), gamma(C);
    for (int c = 0; c < C; ++c) {
        for (int k = 0; k < 12; ++k) beta[c*12+k] = weight[c*21+k];
        for (int k = 0; k < 8;  ++k) alpha[c*8+k]  = weight[c*21+12+k];
        gamma[c] = weight[c*21+20];
    }

    instantmesh::FlexiMesh mesh;
    instantmesh::flexicubes_extract(vdef.data(), sdf.data(), cubes.data(), C,
                                    beta.data(), alpha.data(), gamma.data(), grid_res, mesh);
    std::fprintf(stderr, "mesh: verts=%zu faces=%zu\n", mesh.vertices.size()/3, mesh.faces.size()/3);
    for (int i = 0; i < 6 && i*3+2 < (int)mesh.vertices.size(); ++i)
        std::fprintf(stderr, "cpp vd[%d] = %g %g %g\n", i, mesh.vertices[i*3], mesh.vertices[i*3+1], mesh.vertices[i*3+2]);

    std::string mk = "mkdir -p " + out_dir;
    std::system(mk.c_str());
    {
        FILE * f = std::fopen((out_dir + "/vertices.bin").c_str(), "wb");
        std::fwrite(mesh.vertices.data(), sizeof(float), mesh.vertices.size(), f); std::fclose(f);
        f = std::fopen((out_dir + "/faces.bin").c_str(), "wb");
        std::fwrite(mesh.faces.data(), sizeof(int32_t), mesh.faces.size(), f); std::fclose(f);
    }
    std::printf("wrote %s/vertices.bin and %s/faces.bin\n", out_dir.c_str(), out_dir.c_str());
    return 0;
}