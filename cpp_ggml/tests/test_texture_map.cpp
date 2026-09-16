// Unit tests for UV unwrapping and texture rasterization
// (src/models/texture_map.{hpp,cpp}). No GPU/weights required.
#include <cstdio>
#include <vector>
#include "models/texture_map.hpp"

int main() {
    using namespace instantmesh;
    int fails = 0;

    // 1) uv_unwrap on a flat quad (two triangles) -> single chart, valid UVs.
    {
        float verts[12] = {0,0,0, 1,0,0, 1,1,0, 0,1,0};
        int32_t faces[6] = {0,1,2, 0,2,3};
        std::vector<float> uvs;
        int charts = 0;
        int rc = uv_unwrap(verts, faces, 4, 2, 64, uvs, &charts);
        bool ok = (rc == 0 && charts == 1);
        for (int v = 0; v < 4 && ok; ++v) {
            float u = uvs[2*v], w = uvs[2*v+1];
            if (!(u >= 0.0f && u <= 1.0f && w >= 0.0f && w <= 1.0f)) ok = false;
        }
        printf("%s uv_unwrap quad: charts=%d uvs=[%g,%g][%g,%g][%g,%g][%g,%g]\n",
               ok ? "PASS" : "FAIL", charts,
               uvs[0],uvs[1],uvs[2],uvs[3],uvs[4],uvs[5],uvs[6],uvs[7]);
        if (!ok) ++fails;
    }

    // 2) rasterize a half-atlas triangle -> ~half covered, valid interpolation.
    {
        float verts[9] = {0,0,0, 1,0,0, 0,1,0};
        int32_t faces[3] = {0,1,2};
        float uvs[6] = {0,0, 1,0, 0,1};   // lower-left half of the atlas
        const int res = 8;
        std::vector<float> gb, gb4;
        std::vector<uint8_t> mask;
        rasterize_texture(verts, faces, uvs, 3, 1, res, gb, gb4, mask);

        size_t covered = 0;
        for (auto m : mask) if (m) ++covered;
        bool cov_ok = (covered > 20 && covered < 45);   // ~ res*res/2 = 32
        printf("%s rasterize triangle coverage: covered=%zu/%d (expect ~%d)\n",
               cov_ok ? "PASS" : "FAIL", covered, res*res, res*res/2);
        if (!cov_ok) ++fails;

        // Every covered pixel's interpolated world position must lie inside
        // the source triangle (x>=0, y>=0, x+y<=1).
        bool interp_ok = true;
        for (int p = 0; p < res*res; ++p) if (mask[p]) {
            float x = gb[3*p], y = gb[3*p+1];
            if (x < -1e-3f || y < -1e-3f || x + y > 1.0f + 1e-2f) { interp_ok = false; break; }
        }
        printf("%s rasterize interpolation in-triangle\n", interp_ok ? "PASS" : "FAIL");
        if (!interp_ok) ++fails;
    }

    if (fails) { printf("TEXTURE_MAP: %d FAILURE(S)\n", fails); return 1; }
    printf("TEXTURE_MAP: all tests passed\n");
    return 0;
}
