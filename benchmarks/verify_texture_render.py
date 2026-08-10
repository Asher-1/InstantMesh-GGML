#!/usr/bin/env python3
"""Verify the ggml textured RENDER is correct and consistently oriented by
comparing it against the ggml vertex-color render on the IDENTICAL mesh
geometry (same f16-CUDA run). Vertex colors were validated vs PyTorch net_rgb
at 51 dB, so if the textured render matches the vertex-color render, the
texture map is correct in both content and orientation."""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from render_textured import render_textured, render_ortho  # noqa: E402


def load_vcol(path):
    verts, cols = [], []
    for line in open(path):
        t = line.split()
        if not t: continue
        if t[0] == "v":
            verts.append([float(t[1]), float(t[2]), float(t[3])])
            cols.append([float(t[4]), float(t[5]), float(t[6])])
        elif t[0] == "f":
            pass
    return np.asarray(verts, np.float32), np.asarray(cols, np.float32)


def flat_render(V, F, colors, size=512):
    """Unshaded vertex-color render (matches render_textured's flat texture)."""
    from render_textured import rot, VIEW
    v = V - V.mean(axis=0); v = v / (np.max(np.abs(v)) + 1e-9)
    R = rot(VIEW["elev"], VIEW["azim"]); v = v @ R.T
    half = size / 2.0
    img = np.zeros((size, size, 3), np.float32)
    zbuf = np.full((size, size), np.inf, np.float32)
    xs = v[:,0]*half + half; ys = v[:,1]*half + half; zs = v[:,2]
    for f in F:
        p = xs[f]; q = ys[f]; z = zs[f]
        cc = (colors[f[0]] + colors[f[1]] + colors[f[2]]) / 3.0
        minx = max(0, int(np.floor(p.min()))); maxx = min(size-1, int(np.ceil(p.max())))
        miny = max(0, int(np.floor(q.min()))); maxy = min(size-1, int(np.ceil(q.max())))
        if minx > maxx or miny > maxy: continue
        gx = np.arange(minx, maxx+1, dtype=np.float32)+0.5
        gy = np.arange(miny, maxy+1, dtype=np.float32)+0.5
        GX, GY = np.meshgrid(gx, gy)
        den = (p[1]-p[2])*(q[0]-q[2]) + (p[2]-p[0])*(q[1]-q[2])
        if abs(den) < 1e-12: continue
        l0 = ((p[1]-p[2])*(GX-p[2]) + (p[2]-p[0])*(GY-q[2])) / den
        l1 = ((p[2]-p[0])*(GX-p[0]) + (p[0]-p[2])*(GY-q[0])) / den
        l2 = 1.0-l0-l1
        inside = (l0>=-1e-4)&(l1>=-1e-4)&(l2>=-1e-4)
        if not inside.any(): continue
        dz = l0*z[0]+l1*z[1]+l2*z[2]
        reg = zbuf[miny:maxy+1, minx:maxx+1]
        m = inside & (dz < reg)
        if not m.any(): continue
        reg[m] = dz[m]
        img[miny:maxy+1, minx:maxx+1][m] = cc
    return img


def main():
    V, C = load_vcol("/tmp/ch_vcol.obj")
    faces = []
    for line in open("/tmp/ch_vcol.obj"):
        t = line.split()
        if t and t[0] == "f":
            faces.append([int(x.split("/")[0]) - 1 for x in t[1:4]])
    F = np.asarray(faces, np.int64)

    r_vc = render_ortho(V, F, C)        # shaded (has lighting)
    r_vc_flat = flat_render(V, F, C)    # flat (no lighting, matches textured)
    r_tex = render_textured("/tmp/ch_dump.obj")
    mask = r_vc_flat.sum(axis=2) > 0.01

    def stats(a, b):
        mse = ((a - b) ** 2)[mask].mean()
        psnr = 20*np.log10(1/np.sqrt(mse)) if mse > 0 else float("inf")
        corr = np.corrcoef(a[mask].ravel(), b[mask].ravel())[0, 1]
        return psnr, np.abs(a-b)[mask].mean(), corr

    p1, m1, c1 = stats(r_tex, r_vc_flat)     # textured vs flat vertex
    p2, m2, c2 = stats(r_tex, r_vc)          # textured vs shaded vertex
    print(f"textured vs FLAT vertex-color render (same geometry, {mask.sum()} px):")
    print(f"  PSNR={p1:.2f} dB  MAE={m1:.4f}  r={c1:.4f}")
    print(f"textured vs SHADED vertex-color render: PSNR={p2:.2f} dB  MAE={m2:.4f}  r={c2:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(r_vc_flat); ax[0].set_title("vertex color (flat)"); ax[0].axis("off")
    ax[1].imshow(r_tex); ax[1].set_title("ggml texture map"); ax[1].axis("off")
    ax[2].imshow(np.abs(r_tex - r_vc_flat)); ax[2].set_title(f"|diff| PSNR={p1:.1f}dB"); ax[2].axis("off")
    ax[3].imshow(r_vc); ax[3].set_title("vertex color (shaded)"); ax[3].axis("off")
    fig.tight_layout()
    out = os.path.join(ROOT, "benchmarks", "results", "tex_vs_vertexcolor.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
