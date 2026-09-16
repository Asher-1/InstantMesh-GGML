#!/usr/bin/env python3
"""Clean check: for each covered atlas pixel p, reconstruct the expected world
position from verts.bin + uvs.bin via barycentric interpolation, and compare to
ggml's gb.bin. If they match (MAE~0), the bake world positions are the true
surface -> the texture content is correct and any render mismatch is elsewhere.
No mask-mutation bug here: we process each pixel independently."""
import os
import numpy as np

D = "/tmp/ch_dump"
TR = 1024

uvs = np.fromfile(D + ".uvs.bin", np.float32).reshape(-1, 2)  # (3*nf,2) per-corner atlas UV
verts = np.fromfile(D + ".verts.bin", np.float32).reshape(-1, 3)  # (3*nf,3) split positions
gb = np.fromfile(D + ".gb.bin", np.float32).reshape(TR, TR, 3)
mask = np.fromfile(D + ".mask.bin", np.uint8).reshape(TR, TR)
nc = len(uvs)
nf = nc // 3
# split mesh faces are identity corners {3f, 3f+1, 3f+2}
F = np.arange(nf * 3, dtype=np.int64).reshape(nf, 3)

# We don't have faces here, so instead reconstruct world position at each UV
# pixel using the fact that a covered pixel's gb is a barycentric interpolation.
# build pixel -> face that covers it (same traversal as ggml: first-written wins)
done = np.zeros((TR, TR), bool)
rec = np.zeros((TR, TR, 3), np.float32)
for fi in range(len(F)):
    f = F[fi]
    p = uvs[f, 0] * TR
    q = uvs[f, 1] * TR
    w0, w1, w2 = verts[f[0]], verts[f[1]], verts[f[2]]
    minx = max(0, int(p.min())); maxx = min(TR - 1, int(np.ceil(p.max())))
    miny = max(0, int(q.min())); maxy = min(TR - 1, int(np.ceil(q.max())))
    if minx > maxx or miny > maxy: continue
    gx = np.arange(minx, maxx + 1, dtype=np.float32) + 0.5
    gy = np.arange(miny, maxy + 1, dtype=np.float32) + 0.5
    GX, GY = np.meshgrid(gx, gy)
    den = (p[1] - p[2]) * (q[0] - q[2]) + (p[2] - p[0]) * (q[1] - q[2])
    if abs(den) < 1e-9: continue
    l0 = ((p[1] - p[2]) * (GX - p[2]) + (p[2] - p[0]) * (GY - q[2])) / den
    l1 = ((p[2] - p[0]) * (GX - p[0]) + (p[0] - p[2]) * (GY - q[0])) / den
    l2 = 1.0 - l0 - l1
    inside = (l0 >= -1e-4) & (l1 >= -1e-4) & (l2 >= -1e-4)
    if not inside.any(): continue
    wp = l0[..., None] * w0 + l1[..., None] * w1 + l2[..., None] * w2
    sel = inside & ~done[miny:maxy + 1, minx:maxx + 1]
    rec[miny:maxy + 1, minx:maxx + 1][sel] = wp[sel]
    done[miny:maxy + 1, minx:maxx + 1][sel] = True

m = mask.astype(bool) & done
d = np.abs(gb - rec)[m]
print(f"[gbcheck] ggml gb vs Python surface reconstruction over {m.sum()} px")
print(f"  MAE={d.mean():.5f}  max={d.max():.5f}")
print(f"  ggml mean={gb[m].mean(axis=0).round(4)}  rec mean={rec[m].mean(axis=0).round(4)}")
print(f"  covered_by_python={done.mean():.3f}  ggml_mask={mask.mean():.3f}")
# also: how well does gb sit ON the surface? distance from gb[p] to nearest face
